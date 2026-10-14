//! Images and buffers: NV12 video images with per-plane views, dmabuf
//! import (captured frames) and export (the client's output), and host
//! visible buffers for bitstreams and readback.

use std::os::fd::{AsFd, AsRawFd, BorrowedFd, FromRawFd, OwnedFd};
use std::sync::Arc;

use ash::vk;
use drm_fourcc::DrmFourcc;
use gliff_va::PrimeDescriptor;

use crate::device::Gpu;
use crate::{Error, Result};

pub const NV12: vk::Format = vk::Format::G8_B8R8_2PLANE_420_UNORM;
pub const LINEAR_MODIFIER: u64 = 0;

/// How the display image is used: the recombine writes it, the readback
/// copies from it.
pub(crate) const DISPLAY_USAGE: vk::ImageUsageFlags = vk::ImageUsageFlags::from_raw(
    vk::ImageUsageFlags::STORAGE.as_raw() | vk::ImageUsageFlags::TRANSFER_SRC.as_raw(),
);
/// The display image is BGRA memory the shader stores to through an RGBA
/// view (see recombine.comp).
pub(crate) const DISPLAY_VIEW_FORMATS: [vk::Format; 2] =
    [vk::Format::B8G8R8A8_UNORM, vk::Format::R8G8B8A8_UNORM];

/// Whether a display image with `modifier` can be written and exported,
/// given the driver's modifiers for its BGRA memory and its RGBA view: one
/// memory plane, so one dmabuf describes it, copies out of the BGRA image,
/// and shader stores through the RGBA view.
pub(crate) fn writable_display(
    modifier: u64,
    bgra: &[vk::DrmFormatModifierPropertiesEXT],
    rgba: &[vk::DrmFormatModifierPropertiesEXT],
) -> bool {
    let find = |list: &[vk::DrmFormatModifierPropertiesEXT]| {
        list.iter()
            .find(|p| p.drm_format_modifier == modifier)
            .copied()
    };
    let (Some(bgra), Some(rgba)) = (find(bgra), find(rgba)) else {
        return false;
    };
    bgra.drm_format_modifier_plane_count == 1
        && bgra
            .drm_format_modifier_tiling_features
            .contains(vk::FormatFeatureFlags::TRANSFER_SRC)
        && rgba
            .drm_format_modifier_tiling_features
            .contains(vk::FormatFeatureFlags::STORAGE_IMAGE)
}

/// A dmabuf plane description shared with the capture and display sides.
pub struct DmabufPlane<'a> {
    pub fd: BorrowedFd<'a>,
    pub width: u32,
    pub height: u32,
    pub offset: u32,
    pub stride: u32,
    pub fourcc: DrmFourcc,
    pub modifier: u64,
}

/// A device image plus its memory and the views the pipelines use.
pub struct Image {
    gpu: Arc<Gpu>,
    pub(crate) image: vk::Image,
    /// One allocation, or one per plane for a disjoint image.
    memory: Vec<vk::DeviceMemory>,
    pub format: vk::Format,
    pub width: u32,
    pub height: u32,
    pub layers: u32,
    /// Whole-image view (for video ops), one per array layer.
    pub(crate) layer_views: Vec<vk::ImageView>,
    /// Per-plane views of layer 0 (R8 for Y, R8G8 for UV) for compute.
    pub(crate) plane_views: Vec<vk::ImageView>,
    pub(crate) layout: std::cell::Cell<vk::ImageLayout>,
}

impl Image {
    /// Create an NV12 image the compute shaders read and write through its
    /// plane views, and that copies to and from imported surfaces.
    pub(crate) fn nv12(gpu: &Arc<Gpu>, width: u32, height: u32) -> Result<Self> {
        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC;
        let families = gpu.families.all();
        let mut info = vk::ImageCreateInfo::default()
            .flags(vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE)
            .image_type(vk::ImageType::TYPE_2D)
            .format(NV12)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        if families.len() > 1 {
            info = info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families);
        }
        // SAFETY: valid create info; memory is bound before any use.
        let (image, memory) = unsafe {
            let image = gpu.device.create_image(&info, None)?;
            let reqs = gpu.device.get_image_memory_requirements(image);
            let memory = gpu.allocate(reqs, vk::MemoryPropertyFlags::DEVICE_LOCAL, None)?;
            gpu.device.bind_image_memory(image, memory, 0)?;
            (image, memory)
        };
        let mut img = Self {
            gpu: gpu.clone(),
            image,
            memory: vec![memory],
            format: NV12,
            width,
            height,
            layers: 1,
            layer_views: Vec::new(),
            plane_views: Vec::new(),
            layout: vk::ImageLayout::UNDEFINED.into(),
        };
        let plane_usage = vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED;
        img.plane_views.push(img.view(
            vk::Format::R8_UNORM,
            vk::ImageAspectFlags::PLANE_0,
            0,
            plane_usage,
        )?);
        img.plane_views.push(img.view(
            vk::Format::R8G8_UNORM,
            vk::ImageAspectFlags::PLANE_1,
            0,
            plane_usage,
        )?);
        Ok(img)
    }

    /// Create a BGRA image whose memory is exportable as a dmabuf (the
    /// client's display output), with one of `modifiers`, as the driver
    /// chooses.
    pub(crate) fn exportable_bgra(
        gpu: &Arc<Gpu>,
        width: u32,
        height: u32,
        modifiers: &[u64],
    ) -> Result<Self> {
        let mut mod_list =
            vk::ImageDrmFormatModifierListCreateInfoEXT::default().drm_format_modifiers(modifiers);
        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut format_list =
            vk::ImageFormatListCreateInfo::default().view_formats(&DISPLAY_VIEW_FORMATS);
        let info = vk::ImageCreateInfo::default()
            .flags(vk::ImageCreateFlags::MUTABLE_FORMAT)
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::B8G8R8A8_UNORM)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(DISPLAY_USAGE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut mod_list)
            .push_next(&mut external)
            .push_next(&mut format_list);
        // SAFETY: valid create info; dedicated exportable allocation bound
        // before use.
        let (image, memory) = unsafe {
            let image = gpu.device.create_image(&info, None)?;
            let mut dedicated_reqs = vk::MemoryDedicatedRequirements::default();
            let mut reqs = vk::MemoryRequirements2::default().push_next(&mut dedicated_reqs);
            gpu.device.get_image_memory_requirements2(
                &vk::ImageMemoryRequirementsInfo2::default().image(image),
                &mut reqs,
            );
            let mut export = vk::ExportMemoryAllocateInfo::default()
                .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let mut dedicated = vk::MemoryDedicatedAllocateInfo::default().image(image);
            let alloc = vk::MemoryAllocateInfo::default()
                .allocation_size(reqs.memory_requirements.size)
                .memory_type_index(gpu.memory_type(
                    reqs.memory_requirements.memory_type_bits,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                )?)
                .push_next(&mut export)
                .push_next(&mut dedicated);
            let memory = gpu.device.allocate_memory(&alloc, None)?;
            gpu.device.bind_image_memory(image, memory, 0)?;
            (image, memory)
        };
        let mut img = Self {
            gpu: gpu.clone(),
            image,
            memory: vec![memory],
            format: vk::Format::B8G8R8A8_UNORM,
            width,
            height,
            layers: 1,
            layer_views: Vec::new(),
            plane_views: Vec::new(),
            layout: vk::ImageLayout::UNDEFINED.into(),
        };
        // The shader declares `rgba8`, so it stores through an RGBA view of
        // the BGRA memory with the channels swapped (see recombine.comp).
        img.layer_views.push(img.view(
            vk::Format::R8G8B8A8_UNORM,
            vk::ImageAspectFlags::COLOR,
            0,
            vk::ImageUsageFlags::STORAGE,
        )?);
        Ok(img)
    }

    /// A plain single-plane, optimally tiled device image.
    pub(crate) fn plain(
        gpu: &Arc<Gpu>,
        format: vk::Format,
        width: u32,
        height: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<Self> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        // SAFETY: valid create info; memory is bound before any use.
        let (image, memory) = unsafe {
            let image = gpu.device.create_image(&info, None)?;
            let reqs = gpu.device.get_image_memory_requirements(image);
            let memory = gpu.allocate(reqs, vk::MemoryPropertyFlags::DEVICE_LOCAL, None)?;
            gpu.device.bind_image_memory(image, memory, 0)?;
            (image, memory)
        };
        let mut img = Self {
            gpu: gpu.clone(),
            image,
            memory: vec![memory],
            format,
            width,
            height,
            layers: 1,
            layer_views: Vec::new(),
            plane_views: Vec::new(),
            layout: vk::ImageLayout::UNDEFINED.into(),
        };
        img.layer_views.push(img.view(
            format,
            vk::ImageAspectFlags::COLOR,
            0,
            usage & vk::ImageUsageFlags::SAMPLED,
        )?);
        Ok(img)
    }

    /// A BGRA image filled by a buffer copy, for the upload path.
    pub(crate) fn bgra_upload(gpu: &Arc<Gpu>, width: u32, height: u32) -> Result<Self> {
        Self::plain(
            gpu,
            vk::Format::B8G8R8A8_UNORM,
            width,
            height,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
        )
    }

    /// Export the single-plane image as a dmabuf for another API.
    pub fn export_dmabuf(&self) -> Result<ExportedDmabuf> {
        // SAFETY: the memory was allocated exportable; the fd is owned by us.
        unsafe {
            let info = vk::MemoryGetFdInfoKHR::default()
                .memory(self.memory[0])
                .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
            let fd = self.gpu.external_fd.get_memory_fd(&info)?;
            let fd = OwnedFd::from_raw_fd(fd);
            let mut props = vk::ImageDrmFormatModifierPropertiesEXT::default();
            self.gpu
                .drm_modifier
                .get_image_drm_format_modifier_properties(self.image, &mut props)?;
            let sub = vk::ImageSubresource::default()
                .aspect_mask(vk::ImageAspectFlags::MEMORY_PLANE_0_EXT);
            let layout = self
                .gpu
                .device
                .get_image_subresource_layout(self.image, sub);
            Ok(ExportedDmabuf {
                fd,
                width: self.width,
                height: self.height,
                fourcc: DrmFourcc::Xrgb8888,
                modifier: props.drm_format_modifier,
                offset: layout.offset as u32,
                stride: layout.row_pitch as u32,
            })
        }
    }

    /// Import a captured dmabuf (single-plane 32-bit RGB) for sampling.
    pub(crate) fn import_dmabuf(gpu: &Arc<Gpu>, plane: &DmabufPlane) -> Result<Self> {
        let format = match plane.fourcc {
            DrmFourcc::Xrgb8888 | DrmFourcc::Argb8888 => vk::Format::B8G8R8A8_UNORM,
            DrmFourcc::Xbgr8888 | DrmFourcc::Abgr8888 => vk::Format::R8G8B8A8_UNORM,
            other => return Err(Error::Unsupported(format!("capture format {other:?}"))),
        };
        let layouts = [vk::SubresourceLayout {
            offset: plane.offset as u64,
            size: 0,
            row_pitch: plane.stride as u64,
            array_pitch: 0,
            depth_pitch: 0,
        }];
        let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(plane.modifier)
            .plane_layouts(&layouts);
        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: plane.width,
                height: plane.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(vk::ImageUsageFlags::SAMPLED)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut explicit)
            .push_next(&mut external);
        let (image, memory) = Self::create_imported(gpu, &info, plane.fd)?;
        let mut img = Self {
            gpu: gpu.clone(),
            image,
            memory: vec![memory],
            format,
            width: plane.width,
            height: plane.height,
            layers: 1,
            layer_views: Vec::new(),
            plane_views: Vec::new(),
            layout: vk::ImageLayout::UNDEFINED.into(),
        };
        img.layer_views.push(img.view(
            format,
            vk::ImageAspectFlags::COLOR,
            0,
            vk::ImageUsageFlags::SAMPLED,
        )?);
        Ok(img)
    }

    /// Import an exported VA-API NV12 surface. With `storage` the plane
    /// views can be written by the split shader; without it the image is a
    /// copy target only (and sampled either way).
    pub(crate) fn import_nv12(
        gpu: &Arc<Gpu>,
        desc: &PrimeDescriptor,
        storage: bool,
    ) -> Result<Self> {
        let plane_objects: Vec<usize> = desc.planes.iter().map(|p| p.object).collect();
        let binding = nv12_binding(
            desc.objects.len(),
            &plane_objects,
            desc.modifier,
            gpu.modifier_properties(NV12, desc.modifier),
        )?;
        let layouts: Vec<vk::SubresourceLayout> = desc
            .planes
            .iter()
            .map(|p| vk::SubresourceLayout {
                offset: p.offset as u64,
                size: 0,
                row_pitch: p.pitch as u64,
                array_pitch: 0,
                depth_pitch: 0,
            })
            .collect();
        let mut usage = vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::TRANSFER_DST;
        if storage {
            usage |= vk::ImageUsageFlags::STORAGE;
        }
        let families = gpu.families.all();
        let mut explicit = vk::ImageDrmFormatModifierExplicitCreateInfoEXT::default()
            .drm_format_modifier(desc.modifier)
            .plane_layouts(&layouts);
        let mut external = vk::ExternalMemoryImageCreateInfo::default()
            .handle_types(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT);
        let mut flags = vk::ImageCreateFlags::MUTABLE_FORMAT | vk::ImageCreateFlags::EXTENDED_USAGE;
        if binding == Nv12Binding::Disjoint {
            flags |= vk::ImageCreateFlags::DISJOINT;
        }
        let mut info = vk::ImageCreateInfo::default()
            .flags(flags)
            .image_type(vk::ImageType::TYPE_2D)
            .format(NV12)
            .extent(vk::Extent3D {
                width: desc.width,
                height: desc.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .push_next(&mut explicit)
            .push_next(&mut external);
        if families.len() > 1 {
            info = info
                .sharing_mode(vk::SharingMode::CONCURRENT)
                .queue_family_indices(&families);
        }
        let (image, memory) = match binding {
            Nv12Binding::Single => {
                let (image, memory) = Self::create_imported(gpu, &info, desc.objects[0].as_fd())?;
                (image, vec![memory])
            }
            Nv12Binding::Disjoint => Self::create_imported_disjoint(
                gpu,
                &info,
                [desc.objects[0].as_fd(), desc.objects[1].as_fd()],
            )?,
        };
        let mut img = Self {
            gpu: gpu.clone(),
            image,
            memory,
            format: NV12,
            width: desc.width,
            height: desc.height,
            layers: 1,
            layer_views: Vec::new(),
            plane_views: Vec::new(),
            layout: vk::ImageLayout::GENERAL.into(),
        };
        let plane_usage = usage & (vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED);
        img.plane_views.push(img.view(
            vk::Format::R8_UNORM,
            vk::ImageAspectFlags::PLANE_0,
            0,
            plane_usage,
        )?);
        img.plane_views.push(img.view(
            vk::Format::R8G8_UNORM,
            vk::ImageAspectFlags::PLANE_1,
            0,
            plane_usage,
        )?);
        Ok(img)
    }

    /// Create `info` and bind it to the dmabuf's memory.
    fn create_imported(
        gpu: &Arc<Gpu>,
        info: &vk::ImageCreateInfo,
        fd: BorrowedFd,
    ) -> Result<(vk::Image, vk::DeviceMemory)> {
        // SAFETY: the image is destroyed on every error path, and the memory
        // imported for it is bound before it is returned.
        unsafe {
            let image = gpu.device.create_image(info, None)?;
            let reqs = gpu.device.get_image_memory_requirements(image);
            let memory = match import_memory(gpu, reqs, fd, Some(image)) {
                Ok(m) => m,
                Err(e) => {
                    gpu.device.destroy_image(image, None);
                    return Err(e);
                }
            };
            if let Err(e) = gpu.device.bind_image_memory(image, memory, 0) {
                gpu.device.destroy_image(image, None);
                gpu.device.free_memory(memory, None);
                return Err(e.into());
            }
            Ok((image, memory))
        }
    }

    /// Create the disjoint `info` and bind each memory plane to its own
    /// dmabuf, for drivers that export one object per plane (NVIDIA).
    fn create_imported_disjoint(
        gpu: &Arc<Gpu>,
        info: &vk::ImageCreateInfo,
        fds: [BorrowedFd; 2],
    ) -> Result<(vk::Image, Vec<vk::DeviceMemory>)> {
        const ASPECTS: [vk::ImageAspectFlags; 2] = [
            vk::ImageAspectFlags::MEMORY_PLANE_0_EXT,
            vk::ImageAspectFlags::MEMORY_PLANE_1_EXT,
        ];
        // SAFETY: as in create_imported; the memory imported so far is freed
        // with the image on an error path.
        unsafe {
            let image = gpu.device.create_image(info, None)?;
            let mut memory = Vec::with_capacity(2);
            let bound = (|| -> Result<()> {
                for (fd, aspect) in fds.into_iter().zip(ASPECTS) {
                    let mut plane =
                        vk::ImagePlaneMemoryRequirementsInfo::default().plane_aspect(aspect);
                    let reqs_info = vk::ImageMemoryRequirementsInfo2::default()
                        .image(image)
                        .push_next(&mut plane);
                    let mut reqs = vk::MemoryRequirements2::default();
                    gpu.device
                        .get_image_memory_requirements2(&reqs_info, &mut reqs);
                    // A disjoint image cannot take a dedicated allocation
                    // (VUID-VkMemoryDedicatedAllocateInfo-image-01797).
                    memory.push(import_memory(gpu, reqs.memory_requirements, fd, None)?);
                }
                let mut planes =
                    ASPECTS.map(|a| vk::BindImagePlaneMemoryInfo::default().plane_aspect(a));
                let binds: Vec<vk::BindImageMemoryInfo> = planes
                    .iter_mut()
                    .zip(&memory)
                    .map(|(plane, &m)| {
                        vk::BindImageMemoryInfo::default()
                            .image(image)
                            .memory(m)
                            .push_next(plane)
                    })
                    .collect();
                gpu.device.bind_image_memory2(&binds)?;
                Ok(())
            })();
            if let Err(e) = bound {
                gpu.device.destroy_image(image, None);
                for &m in &memory {
                    gpu.device.free_memory(m, None);
                }
                return Err(e);
            }
            Ok((image, memory))
        }
    }

    /// Record a plane-by-plane copy of another NV12 image of the same size.
    /// `src` must be in TRANSFER_SRC_OPTIMAL and `self` in TRANSFER_DST_OPTIMAL.
    pub(crate) fn copy_nv12_from(&self, cmd: vk::CommandBuffer, src: &Image) {
        let region = |aspect: vk::ImageAspectFlags, w: u32, h: u32| vk::ImageCopy {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: aspect,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_offset: vk::Offset3D::default(),
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: aspect,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_offset: vk::Offset3D::default(),
            extent: vk::Extent3D {
                width: w,
                height: h,
                depth: 1,
            },
        };
        let (w, h) = (self.width.min(src.width), self.height.min(src.height));
        let regions = [
            region(vk::ImageAspectFlags::PLANE_0, w, h),
            region(vk::ImageAspectFlags::PLANE_1, w / 2, h / 2),
        ];
        // SAFETY: recording into a command buffer in the recording state;
        // both images are NV12 with memory bound.
        unsafe {
            self.gpu.device.cmd_copy_image(
                cmd,
                src.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }
    }

    /// A view limited to `usage`, which must be a subset of the image's usage:
    /// plane views carry only the compute usages, whole-image views only the
    /// video ones, since the plane formats and NV12 support different features.
    fn view(
        &self,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
        layer: u32,
        usage: vk::ImageUsageFlags,
    ) -> Result<vk::ImageView> {
        let mut usage_info = vk::ImageViewUsageCreateInfo::default().usage(usage);
        let info = vk::ImageViewCreateInfo::default()
            .image(self.image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: aspect,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: layer,
                layer_count: 1,
            })
            .push_next(&mut usage_info);
        // SAFETY: valid image and range.
        Ok(unsafe { self.gpu.device.create_image_view(&info, None) }?)
    }

    pub(crate) fn view0(&self) -> vk::ImageView {
        self.layer_views[0]
    }

    /// Record a layout transition for all layers, with a full barrier.
    pub(crate) fn transition(&self, cmd: vk::CommandBuffer, new: vk::ImageLayout) {
        let old = self.layout.get();
        if old == new {
            return;
        }
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
            .old_layout(old)
            .new_layout(new)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: self.layers,
            });
        let barriers = [barrier];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        // SAFETY: recording into a command buffer in the recording state.
        unsafe { self.gpu.device.cmd_pipeline_barrier2(cmd, &dep) };
        self.layout.set(new);
    }

    /// Take the image back from another API (VA-API), with its contents
    /// kept, and move it to `new` for this queue. Imported surfaces start
    /// out owned by that API, in GENERAL layout.
    pub(crate) fn acquire_foreign(&self, cmd: vk::CommandBuffer, new: vk::ImageLayout) {
        self.foreign_barrier(cmd, self.gpu.foreign_family, self.gpu.families.compute, new);
    }

    /// Take an exported display image back from the display side (the
    /// compositor) before using it, in GENERAL layout; it is handed out again
    /// with [`Self::release_foreign`]. A fresh image was never handed out.
    pub(crate) fn acquire_display(&self, cmd: vk::CommandBuffer) {
        if self.layout.get() == vk::ImageLayout::UNDEFINED {
            self.transition(cmd, vk::ImageLayout::GENERAL);
        } else {
            self.acquire_foreign(cmd, vk::ImageLayout::GENERAL);
        }
    }

    /// Hand the image to another API (VA-API, or a compositor for a display
    /// image) to read, contents kept.
    pub(crate) fn release_foreign(&self, cmd: vk::CommandBuffer) {
        self.foreign_barrier(
            cmd,
            self.gpu.families.compute,
            self.gpu.foreign_family,
            vk::ImageLayout::GENERAL,
        );
    }

    fn foreign_barrier(&self, cmd: vk::CommandBuffer, src: u32, dst: u32, new: vk::ImageLayout) {
        let barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE | vk::AccessFlags2::MEMORY_READ)
            .old_layout(self.layout.get())
            .new_layout(new)
            .src_queue_family_index(src)
            .dst_queue_family_index(dst)
            .image(self.image)
            .subresource_range(vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: self.layers,
            });
        let barriers = [barrier];
        let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
        // SAFETY: recording into a command buffer in the recording state.
        unsafe { self.gpu.device.cmd_pipeline_barrier2(cmd, &dep) };
        self.layout.set(new);
    }

    /// Full memory barrier without a layout change (between a write and a
    /// read of the same image in the same layout).
    pub(crate) fn memory_barrier(&self, cmd: vk::CommandBuffer) {
        let barrier = vk::MemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
        let barriers = [barrier];
        let dep = vk::DependencyInfo::default().memory_barriers(&barriers);
        // SAFETY: recording into a command buffer in the recording state.
        unsafe { self.gpu.device.cmd_pipeline_barrier2(cmd, &dep) };
    }

    /// Record a copy of packed 32-bit pixels from `buf` into the image.
    pub(crate) fn copy_rgba_from_buffer(&self, cmd: vk::CommandBuffer, buf: &HostBuffer) {
        let regions = [vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: self.width,
            buffer_image_height: self.height,
            image_subresource: layers(vk::ImageAspectFlags::COLOR, 0),
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width: self.width,
                height: self.height,
                depth: 1,
            },
        }];
        // SAFETY: valid regions within both the image and the buffer.
        unsafe {
            self.gpu.device.cmd_copy_buffer_to_image(
                cmd,
                buf.buffer,
                self.image,
                self.layout.get(),
                &regions,
            )
        };
    }

    /// Record a copy of a single-plane 32-bit image into `buf`, packed.
    pub(crate) fn copy_rgba_to_buffer(&self, cmd: vk::CommandBuffer, buf: &HostBuffer) {
        let regions = [vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: self.width,
            buffer_image_height: self.height,
            image_subresource: layers(vk::ImageAspectFlags::COLOR, 0),
            image_offset: vk::Offset3D::default(),
            image_extent: vk::Extent3D {
                width: self.width,
                height: self.height,
                depth: 1,
            },
        }];
        // SAFETY: valid regions within both the image and the buffer.
        unsafe {
            self.gpu.device.cmd_copy_image_to_buffer(
                cmd,
                self.image,
                self.layout.get(),
                buf.buffer,
                &regions,
            )
        };
    }
}

fn layers(aspect: vk::ImageAspectFlags, layer: u32) -> vk::ImageSubresourceLayers {
    vk::ImageSubresourceLayers {
        aspect_mask: aspect,
        mip_level: 0,
        base_array_layer: layer,
        layer_count: 1,
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        // SAFETY: owners wait for GPU work that uses the image before dropping.
        unsafe {
            for v in self.plane_views.drain(..).chain(self.layer_views.drain(..)) {
                self.gpu.device.destroy_image_view(v, None);
            }
            self.gpu.device.destroy_image(self.image, None);
            for &m in &self.memory {
                self.gpu.device.free_memory(m, None);
            }
        }
    }
}

/// A dmabuf handed to another API (GTK), describing one plane.
#[derive(Debug)]
pub struct ExportedDmabuf {
    pub fd: OwnedFd,
    pub width: u32,
    pub height: u32,
    pub fourcc: DrmFourcc,
    pub modifier: u64,
    pub offset: u32,
    pub stride: u32,
}

fn libc_dup(fd: BorrowedFd) -> Result<OwnedFd> {
    Ok(fd.try_clone_to_owned()?)
}

/// How an exported NV12 surface's planes bind to Vulkan memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Nv12Binding {
    /// Both planes in one dmabuf object: one import for the whole image.
    Single,
    /// Each plane in its own object (NVIDIA): a disjoint image with one
    /// import per plane.
    Disjoint,
}

/// Choose how to import a surface exported as `objects` dmabuf objects, its
/// planes in `plane_objects`, given the driver's `props` for `modifier`.
fn nv12_binding(
    objects: usize,
    plane_objects: &[usize],
    modifier: u64,
    props: Option<vk::DrmFormatModifierPropertiesEXT>,
) -> Result<Nv12Binding> {
    if plane_objects.len() != 2 {
        return Err(Error::Unsupported(format!(
            "surface exported with {} planes; NV12 has two",
            plane_objects.len()
        )));
    }
    let props = props.ok_or_else(|| {
        Error::Unsupported(format!(
            "NV12 with modifier {modifier:#x} is not importable"
        ))
    })?;
    let memory_planes = props.drm_format_modifier_plane_count;
    if memory_planes != 2 {
        return Err(Error::Unsupported(format!(
            "NV12 with modifier {modifier:#x} has {memory_planes} memory planes"
        )));
    }
    match objects {
        1 => Ok(Nv12Binding::Single),
        2 if plane_objects == [0, 1] => {
            if props
                .drm_format_modifier_tiling_features
                .contains(vk::FormatFeatureFlags::DISJOINT)
            {
                Ok(Nv12Binding::Disjoint)
            } else {
                Err(Error::Unsupported(format!(
                    "NV12 with modifier {modifier:#x} cannot bind its planes to separate dmabuf objects"
                )))
            }
        }
        n => Err(Error::Unsupported(format!(
            "surface exported as {n} dmabuf objects; one, or one per plane in order, is supported"
        ))),
    }
}

/// Import `fd` as memory that meets `reqs`, dedicated to `image` if given.
///
/// # Safety
///
/// `image`, when given, must be a valid image this memory will be bound to.
unsafe fn import_memory(
    gpu: &Gpu,
    reqs: vk::MemoryRequirements,
    fd: BorrowedFd,
    dedicated: Option<vk::Image>,
) -> Result<vk::DeviceMemory> {
    // SAFETY: the fd is duplicated for Vulkan, which takes ownership of the
    // duplicate on a successful import.
    unsafe {
        let mut fd_props = vk::MemoryFdPropertiesKHR::default();
        gpu.external_fd.get_memory_fd_properties(
            vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT,
            fd.as_raw_fd(),
            &mut fd_props,
        )?;
        let dup = libc_dup(fd)?;
        let type_bits = reqs.memory_type_bits & fd_props.memory_type_bits;
        let type_index = gpu.memory_type(type_bits, vk::MemoryPropertyFlags::empty())?;
        let mut import = vk::ImportMemoryFdInfoKHR::default()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(dup.as_raw_fd());
        let mut dedicated_info = vk::MemoryDedicatedAllocateInfo::default();
        let mut alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(reqs.size)
            .memory_type_index(type_index)
            .push_next(&mut import);
        if let Some(image) = dedicated {
            dedicated_info = dedicated_info.image(image);
            alloc = alloc.push_next(&mut dedicated_info);
        }
        let memory = gpu.device.allocate_memory(&alloc, None)?;
        std::mem::forget(dup);
        Ok(memory)
    }
}

/// A host-visible, persistently mapped buffer.
pub struct HostBuffer {
    gpu: Arc<Gpu>,
    pub(crate) buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    ptr: *mut u8,
    pub size: usize,
}

impl HostBuffer {
    pub(crate) fn new(gpu: &Arc<Gpu>, size: usize, usage: vk::BufferUsageFlags) -> Result<Self> {
        let info = vk::BufferCreateInfo::default()
            .size(size as u64)
            .usage(usage);
        // SAFETY: valid create info; the mapping lives as long as the buffer.
        let (buffer, memory, ptr) = unsafe {
            let buffer = gpu.device.create_buffer(&info, None)?;
            let reqs = gpu.device.get_buffer_memory_requirements(buffer);
            // Cached host memory makes CPU reads (bitstream, readback) fast;
            // fall back to write-combined if the type bits do not allow it.
            let base =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let memory = gpu
                .allocate(reqs, base | vk::MemoryPropertyFlags::HOST_CACHED, None)
                .or_else(|_| gpu.allocate(reqs, base, None))?;
            gpu.device.bind_buffer_memory(buffer, memory, 0)?;
            let ptr = gpu
                .device
                .map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())?
                .cast::<u8>();
            (buffer, memory, ptr)
        };
        Ok(Self {
            gpu: gpu.clone(),
            buffer,
            memory,
            ptr,
            size,
        })
    }

    pub fn write(&self, offset: usize, data: &[u8]) {
        assert!(
            offset <= self.size && data.len() <= self.size - offset,
            "write past the buffer"
        );
        // SAFETY: in bounds of a live host-coherent mapping.
        unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.add(offset), data.len()) };
    }

    pub fn read(&self, offset: usize, len: usize) -> Vec<u8> {
        assert!(
            offset <= self.size && len <= self.size - offset,
            "read past the buffer"
        );
        // SAFETY: in bounds of a live host-coherent mapping; the GPU work
        // that wrote it has completed (caller waited on its fence).
        unsafe { std::slice::from_raw_parts(self.ptr.add(offset), len).to_vec() }
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        // SAFETY: owners wait for GPU work that uses the buffer before dropping.
        unsafe {
            self.gpu.device.unmap_memory(self.memory);
            self.gpu.device.destroy_buffer(self.buffer, None);
            self.gpu.device.free_memory(self.memory, None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const NVIDIA_BLOCK_LINEAR: u64 = 0x0300_0000_0060_6014;

    fn props(
        memory_planes: u32,
        features: vk::FormatFeatureFlags,
    ) -> vk::DrmFormatModifierPropertiesEXT {
        vk::DrmFormatModifierPropertiesEXT {
            drm_format_modifier: NVIDIA_BLOCK_LINEAR,
            drm_format_modifier_plane_count: memory_planes,
            drm_format_modifier_tiling_features: features,
        }
    }

    fn sampled() -> vk::FormatFeatureFlags {
        vk::FormatFeatureFlags::SAMPLED_IMAGE
    }

    fn disjoint() -> vk::FormatFeatureFlags {
        vk::FormatFeatureFlags::SAMPLED_IMAGE | vk::FormatFeatureFlags::DISJOINT
    }

    #[test]
    fn one_object_surface_takes_the_single_import() {
        for features in [sampled(), disjoint()] {
            let binding = nv12_binding(1, &[0, 0], NVIDIA_BLOCK_LINEAR, Some(props(2, features)));
            assert_eq!(binding.unwrap(), Nv12Binding::Single);
        }
    }

    #[test]
    fn one_object_per_plane_with_a_disjoint_modifier_imports_each_plane() {
        let binding = nv12_binding(2, &[0, 1], NVIDIA_BLOCK_LINEAR, Some(props(2, disjoint())));
        assert_eq!(binding.unwrap(), Nv12Binding::Disjoint);
    }

    #[test]
    fn one_object_per_plane_without_disjoint_support_is_refused() {
        let binding = nv12_binding(2, &[0, 1], NVIDIA_BLOCK_LINEAR, Some(props(2, sampled())));
        assert!(matches!(binding, Err(Error::Unsupported(_))), "{binding:?}");
    }

    #[test]
    fn unsupported_layouts_are_refused() {
        let p = Some(props(2, disjoint()));
        for (objects, planes) in [(3, &[0, 1][..]), (2, &[1, 0]), (2, &[0, 0]), (1, &[0])] {
            let binding = nv12_binding(objects, planes, NVIDIA_BLOCK_LINEAR, p);
            assert!(
                matches!(binding, Err(Error::Unsupported(_))),
                "{objects} objects, planes {planes:?}: {binding:?}"
            );
        }
        let one_memory_plane = nv12_binding(1, &[0, 0], 0, Some(props(1, sampled())));
        assert!(matches!(one_memory_plane, Err(Error::Unsupported(_))));
        let unknown_modifier = nv12_binding(2, &[0, 1], NVIDIA_BLOCK_LINEAR, None);
        assert!(matches!(unknown_modifier, Err(Error::Unsupported(_))));
    }

    fn display(
        features: vk::FormatFeatureFlags,
        planes: u32,
    ) -> vk::DrmFormatModifierPropertiesEXT {
        vk::DrmFormatModifierPropertiesEXT {
            drm_format_modifier: NVIDIA_BLOCK_LINEAR,
            drm_format_modifier_plane_count: planes,
            drm_format_modifier_tiling_features: features,
        }
    }

    fn written() -> vk::FormatFeatureFlags {
        vk::FormatFeatureFlags::STORAGE_IMAGE | vk::FormatFeatureFlags::TRANSFER_SRC
    }

    #[test]
    fn a_tiled_display_image_the_shader_writes_is_kept() {
        let (bgra, rgba) = ([display(written(), 1)], [display(written(), 1)]);
        assert!(writable_display(NVIDIA_BLOCK_LINEAR, &bgra, &rgba));
    }

    #[test]
    fn a_display_modifier_with_a_second_memory_plane_is_refused() {
        let (bgra, rgba) = ([display(written(), 2)], [display(written(), 2)]);
        assert!(!writable_display(NVIDIA_BLOCK_LINEAR, &bgra, &rgba));
    }

    #[test]
    fn a_display_modifier_the_shader_cannot_store_to_is_refused() {
        let bgra = [display(written(), 1)];
        let rgba = [display(sampled(), 1)];
        assert!(!writable_display(NVIDIA_BLOCK_LINEAR, &bgra, &rgba));
        let unreadable = [display(vk::FormatFeatureFlags::STORAGE_IMAGE, 1)];
        assert!(!writable_display(NVIDIA_BLOCK_LINEAR, &unreadable, &bgra));
    }

    #[test]
    fn a_display_modifier_the_driver_does_not_list_is_refused() {
        let (bgra, rgba) = ([display(written(), 1)], [display(written(), 1)]);
        assert!(!writable_display(LINEAR_MODIFIER, &bgra, &rgba));
        assert!(!writable_display(NVIDIA_BLOCK_LINEAR, &bgra, &[]));
    }
}
