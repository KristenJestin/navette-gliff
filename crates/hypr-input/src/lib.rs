//! Input injection into Hyprland: virtual keyboard (with an xkb state that
//! derives modifiers from key events) and virtual pointer bound to one output.
//!
//! Runs on its own thread; the owner sends [`InputCmd`]s through [`Input`].

#![forbid(unsafe_code)]

mod clipboard;
mod keymap;
mod keymap_watch;
mod thread;

use std::sync::mpsc;

use calloop::channel::Sender;

pub use clipboard::{Clipboard, ClipboardEvent, ClipboardSink};
pub use hypr_wl::{OutputInfo, Target};
pub use keymap::{key_name, keymap_from_names, KeymapNames};
pub use keymap_watch::{watch_keymap, KeymapSink};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Wayland(#[from] hypr_wl::Error),
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("xkb: {0}")]
    Xkb(String),
    #[error("input: {0}")]
    Input(String),
    #[error("input thread is gone")]
    ThreadGone,
}

impl From<wayland_client::DispatchError> for Error {
    fn from(e: wayland_client::DispatchError) -> Self {
        Error::Wayland(hypr_wl::Error::Wayland(e.to_string()))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    Vertical,
    Horizontal,
}

#[derive(Debug, Clone)]
pub struct InputConfig {
    pub target: Target,
    /// Output the pointer is bound to; absolute motion is in its logical space.
    pub output: String,
    /// Full xkb keymap text (format v1). `None` uses the compositor default.
    pub keymap: Option<String>,
}

impl InputConfig {
    pub fn new(output: impl Into<String>) -> Self {
        Self {
            target: Target::default(),
            output: output.into(),
            keymap: None,
        }
    }
}

#[derive(Debug)]
pub enum InputCmd {
    /// `code` is an evdev keycode (xkb keycode minus 8).
    Key {
        code: u32,
        pressed: bool,
    },
    /// Absolute pointer position in logical output coordinates.
    Motion {
        x: f64,
        y: f64,
    },
    /// Relative pointer motion, in the same units as `Motion`. A client
    /// that locks or confines the pointer (a 3D view's orbit, a game) sees
    /// only relative motion: an absolute `Motion` sends it none. Send it
    /// before the `Motion` that ends at the same place.
    Relative {
        dx: f64,
        dy: f64,
    },
    /// `button` is an evdev `BTN_*` code.
    Button {
        button: u32,
        pressed: bool,
    },
    Axis {
        axis: Axis,
        value: f64,
        discrete: Option<i32>,
        stop: bool,
    },
    SetKeymap(String),
    /// Extent of the output's logical space changed (resize).
    SetExtent {
        width: u32,
        height: u32,
    },
    /// Lock exactly these modifiers, by xkb name (`Lock` for Caps Lock,
    /// `Mod2` for Num Lock), to match another keyboard's locks.
    SetLocked(Vec<String>),
    /// Every key and button held goes up; the locks stay.
    ReleaseAll,
    Stop,
}

#[derive(Debug)]
pub enum InputEvent {
    Ready,
    Error(String),
}

pub type EventSink = Box<dyn FnMut(InputEvent) + Send>;

pub struct Input {
    cmd: Sender<InputCmd>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl Input {
    pub fn start(config: InputConfig, sink: EventSink) -> Result<Self> {
        let (ready_tx, ready_rx) = mpsc::channel();
        let (cmd, join) = thread::spawn(config, sink, ready_tx)?;
        match ready_rx.recv() {
            Ok(Ok(())) => Ok(Self {
                cmd,
                join: Some(join),
            }),
            Ok(Err(e)) => Err(e),
            Err(_) => Err(Error::ThreadGone),
        }
    }

    pub fn send(&self, cmd: InputCmd) -> Result<()> {
        self.cmd.send(cmd).map_err(|_| Error::ThreadGone)
    }

    pub fn key(&self, code: u32, pressed: bool) -> Result<()> {
        self.send(InputCmd::Key { code, pressed })
    }

    pub fn motion(&self, x: f64, y: f64) -> Result<()> {
        self.send(InputCmd::Motion { x, y })
    }

    pub fn button(&self, button: u32, pressed: bool) -> Result<()> {
        self.send(InputCmd::Button { button, pressed })
    }

    pub fn release_all(&self) -> Result<()> {
        self.send(InputCmd::ReleaseAll)
    }
}

impl Drop for Input {
    fn drop(&mut self) {
        let _ = self.cmd.send(InputCmd::ReleaseAll);
        let _ = self.cmd.send(InputCmd::Stop);
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

/// Protocol names the input path needs, for probing.
pub const REQUIRED_GLOBALS: &[&str] = &[
    "zwp_virtual_keyboard_manager_v1",
    "zwlr_virtual_pointer_manager_v1",
];

/// Evdev codes for a handful of keys, for smoke tests.
pub mod keys {
    pub const KEY_E: u32 = 18;
    pub const KEY_H: u32 = 35;
    pub const KEY_L: u32 = 38;
    pub const KEY_O: u32 = 24;
    pub const KEY_SPACE: u32 = 57;
    pub const KEY_ENTER: u32 = 28;
    pub const BTN_LEFT: u32 = 0x110;
    pub const BTN_RIGHT: u32 = 0x111;
    pub const BTN_MIDDLE: u32 = 0x112;
}
