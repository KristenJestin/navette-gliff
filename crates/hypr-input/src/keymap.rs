//! xkb keymap construction and the per-session key state.

use xkbcommon::xkb;

use crate::{Error, Result};

#[derive(Debug, Clone, Default)]
pub struct KeymapNames {
    pub rules: String,
    pub model: String,
    pub layout: String,
    pub variant: String,
    pub options: Option<String>,
}

/// Compile a keymap from RMLVO names and return its text (format v1).
pub fn keymap_from_names(names: &KeymapNames) -> Result<String> {
    let ctx = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
    let keymap = xkb::Keymap::new_from_names(
        &ctx,
        &names.rules,
        &names.model,
        &names.layout,
        &names.variant,
        names.options.clone(),
        xkb::KEYMAP_COMPILE_NO_FLAGS,
    )
    .ok_or_else(|| Error::Xkb(format!("cannot compile keymap {names:?}")))?;
    Ok(keymap.get_as_string(xkb::KEYMAP_FORMAT_TEXT_V1))
}

/// Name of the keysym `evdev_code` gives on its first level in the first
/// layout, e.g. `Control_L`.
pub fn key_name(text: &str, evdev_code: u32) -> Result<String> {
    let keymap = KeyState::from_text(text)?.keymap;
    let syms = keymap.key_get_syms_by_level(xkb::Keycode::new(evdev_code + 8), 0, 0);
    Ok(syms
        .first()
        .map_or_else(String::new, |s| xkb::keysym_get_name(*s)))
}

/// Tracks modifier state for the virtual keyboard from raw key events.
pub struct KeyState {
    pub keymap: xkb::Keymap,
    state: xkb::State,
    pub text: String,
    last_mods: (u32, u32, u32, u32),
}

impl KeyState {
    pub fn from_text(text: &str) -> Result<Self> {
        let ctx = xkb::Context::new(xkb::CONTEXT_NO_FLAGS);
        let keymap = xkb::Keymap::new_from_string(
            &ctx,
            text.to_owned(),
            xkb::KEYMAP_FORMAT_TEXT_V1,
            xkb::KEYMAP_COMPILE_NO_FLAGS,
        )
        .ok_or_else(|| Error::Xkb("cannot compile keymap text".into()))?;
        let state = xkb::State::new(&keymap);
        Ok(Self {
            keymap,
            state,
            text: text.to_owned(),
            last_mods: (0, 0, 0, 0),
        })
    }

    pub fn default_us() -> Result<Self> {
        let text = keymap_from_names(&KeymapNames {
            layout: "us".into(),
            ..Default::default()
        })?;
        Self::from_text(&text)
    }

    /// Apply a key event. Returns the new `(depressed, latched, locked, group)`
    /// when it changed.
    pub fn update(&mut self, evdev_code: u32, pressed: bool) -> Option<(u32, u32, u32, u32)> {
        let dir = if pressed {
            xkb::KeyDirection::Down
        } else {
            xkb::KeyDirection::Up
        };
        self.state
            .update_key(xkb::Keycode::new(evdev_code.saturating_add(8)), dir);
        let mods = self.mods();
        if mods != self.last_mods {
            self.last_mods = mods;
            Some(mods)
        } else {
            None
        }
    }

    /// Take over the state of `old`, a different keymap: its locked layout,
    /// the keys still `held`, then its locked modifiers (matched by name).
    /// Returns the resulting `(depressed, latched, locked, group)`.
    pub fn carry_over(
        &mut self,
        old: &KeyState,
        held: impl IntoIterator<Item = u32>,
    ) -> (u32, u32, u32, u32) {
        let layout = old.state.serialize_layout(xkb::STATE_LAYOUT_LOCKED);
        let layout = if layout < self.keymap.num_layouts() {
            layout
        } else {
            0
        };
        // Held keys are replayed in the carried layout, and before the
        // locked modifiers are set, so a held lock key does not count them
        // as its own and clear them on release.
        self.state.update_mask(0, 0, 0, 0, 0, layout);
        for code in held {
            self.state.update_key(
                xkb::Keycode::new(code.saturating_add(8)),
                xkb::KeyDirection::Down,
            );
        }
        let old_locked = old.state.serialize_mods(xkb::STATE_MODS_LOCKED);
        let mut locked = 0;
        for i in 0..old.keymap.num_mods() {
            if old_locked & (1 << i) == 0 {
                continue;
            }
            let index = self.keymap.mod_get_index(old.keymap.mod_get_name(i));
            if index != xkb::MOD_INVALID {
                locked |= 1 << index;
            }
        }
        self.state.update_mask(
            self.state.serialize_mods(xkb::STATE_MODS_DEPRESSED),
            self.state.serialize_mods(xkb::STATE_MODS_LATCHED),
            locked,
            self.state.serialize_layout(xkb::STATE_LAYOUT_DEPRESSED),
            self.state.serialize_layout(xkb::STATE_LAYOUT_LATCHED),
            layout,
        );
        self.last_mods = self.mods();
        self.last_mods
    }

    fn mods(&self) -> (u32, u32, u32, u32) {
        (
            self.state.serialize_mods(xkb::STATE_MODS_DEPRESSED),
            self.state.serialize_mods(xkb::STATE_MODS_LATCHED),
            self.state.serialize_mods(xkb::STATE_MODS_LOCKED),
            self.state.serialize_layout(xkb::STATE_LAYOUT_EFFECTIVE),
        )
    }

    /// Release the `held` keys: what they hold down goes up, what they
    /// locked (Caps Lock, Num Lock) stays locked. Returns the resulting
    /// `(depressed, latched, locked, group)`.
    pub fn release(&mut self, held: impl IntoIterator<Item = u32>) -> (u32, u32, u32, u32) {
        for code in held {
            self.state.update_key(
                xkb::Keycode::new(code.saturating_add(8)),
                xkb::KeyDirection::Up,
            );
        }
        self.last_mods = self.mods();
        self.last_mods
    }

    /// Lock exactly the modifiers named (xkb names: `Lock` for Caps Lock,
    /// `Mod2` for Num Lock) that this keymap has, and unlock the others;
    /// what is held and the layout stay. Returns the resulting
    /// `(depressed, latched, locked, group)`.
    pub fn set_locked(&mut self, names: &[String]) -> (u32, u32, u32, u32) {
        let mut locked = 0;
        for name in names {
            let index = self.keymap.mod_get_index(name);
            if index != xkb::MOD_INVALID && index < 32 {
                locked |= 1 << index;
            }
        }
        self.state.update_mask(
            self.state.serialize_mods(xkb::STATE_MODS_DEPRESSED),
            self.state.serialize_mods(xkb::STATE_MODS_LATCHED),
            locked,
            self.state.serialize_layout(xkb::STATE_LAYOUT_DEPRESSED),
            self.state.serialize_layout(xkb::STATE_LAYOUT_LATCHED),
            self.state.serialize_layout(xkb::STATE_LAYOUT_LOCKED),
        );
        self.last_mods = self.mods();
        self.last_mods
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_us_keymap_text() {
        let text = keymap_from_names(&KeymapNames {
            layout: "us".into(),
            ..Default::default()
        })
        .unwrap();
        assert!(text.contains("xkb_keymap"));
        assert!(text.contains("xkb_symbols"));
    }

    #[test]
    fn shift_changes_modifiers() {
        let mut ks = KeyState::default_us().unwrap();
        const KEY_LEFTSHIFT: u32 = 42;
        let mods = ks
            .update(KEY_LEFTSHIFT, true)
            .expect("shift press changes mods");
        assert_ne!(mods.0, 0);
        assert!(ks.update(35, true).is_none());
        assert!(ks.update(35, false).is_none());
        let mods = ks
            .update(KEY_LEFTSHIFT, false)
            .expect("shift release changes mods");
        assert_eq!(mods.0, 0);
    }

    #[test]
    fn uploaded_keymap_options_apply() {
        let text = keymap_from_names(&KeymapNames {
            layout: "us".into(),
            options: Some("ctrl:nocaps".into()),
            ..Default::default()
        })
        .unwrap();
        let mut ks = KeyState::from_text(&text).unwrap();
        let control = 1 << ks.keymap.mod_get_index(xkb::MOD_NAME_CTRL);
        const KEY_CAPSLOCK: u32 = 58;
        let mods = ks
            .update(KEY_CAPSLOCK, true)
            .expect("caps press changes mods");
        assert_eq!(mods.0, control);
        assert_eq!(mods.2, 0);
    }

    #[test]
    fn release_lets_go_of_held_keys_and_keeps_the_locks() {
        const KEY_CAPSLOCK: u32 = 58;
        const KEY_LEFTSHIFT: u32 = 42;
        let mut ks = KeyState::default_us().unwrap();
        ks.update(KEY_CAPSLOCK, true);
        ks.update(KEY_CAPSLOCK, false);
        ks.update(KEY_LEFTSHIFT, true);
        let lock = 1 << ks.keymap.mod_get_index(xkb::MOD_NAME_CAPS);
        let (depressed, _, locked, _) = ks.release([KEY_LEFTSHIFT]);
        assert_eq!((depressed, locked), (0, lock));
    }

    #[test]
    fn locks_are_set_by_name() {
        let mut ks = KeyState::default_us().unwrap();
        let lock = 1 << ks.keymap.mod_get_index(xkb::MOD_NAME_CAPS);
        let num = 1 << ks.keymap.mod_get_index(xkb::MOD_NAME_NUM);
        let names = [
            "Lock".to_string(),
            "Mod2".to_string(),
            "NoSuchMod".to_string(),
        ];
        assert_eq!(ks.set_locked(&names).2, lock | num);
        assert_eq!(ks.set_locked(&["Mod2".to_string()]).2, num);
        assert_eq!(ks.set_locked(&[]).2, 0);
    }

    #[test]
    fn carry_over_keeps_locks_and_held_keys() {
        const KEY_CAPSLOCK: u32 = 58;
        const KEY_LEFTCTRL: u32 = 29;
        let mut old = KeyState::default_us().unwrap();
        old.update(KEY_CAPSLOCK, true);
        old.update(KEY_CAPSLOCK, false);
        old.update(KEY_LEFTCTRL, true);
        let text = keymap_from_names(&KeymapNames {
            layout: "us,de".into(),
            ..Default::default()
        })
        .unwrap();
        let mut new = KeyState::from_text(&text).unwrap();
        let (depressed, _, locked, _) = new.carry_over(&old, [KEY_LEFTCTRL]);
        let lock = 1 << new.keymap.mod_get_index(xkb::MOD_NAME_CAPS);
        let control = 1 << new.keymap.mod_get_index(xkb::MOD_NAME_CTRL);
        assert_eq!(locked, lock);
        assert_eq!(depressed, control);
        let mods = new.update(KEY_LEFTCTRL, false).expect("ctrl release");
        assert_eq!((mods.0, mods.2), (0, lock));
    }

    #[test]
    fn carry_over_replays_held_keys_in_the_locked_layout() {
        const KEY_RIGHTALT: u32 = 100;
        let text = keymap_from_names(&KeymapNames {
            layout: "us,de".into(),
            ..Default::default()
        })
        .unwrap();
        let mut old = KeyState::from_text(&text).unwrap();
        old.state.update_mask(0, 0, 0, 0, 0, 1);
        old.update(KEY_RIGHTALT, true);
        let mut new = KeyState::from_text(&text).unwrap();
        let (depressed, _, _, group) = new.carry_over(&old, [KEY_RIGHTALT]);
        let level3 = 1 << new.keymap.mod_get_index("Mod5");
        assert_eq!(group, 1);
        assert_eq!(depressed, level3);
    }

    #[test]
    fn carry_over_keeps_the_lock_of_a_held_lock_key() {
        const KEY_CAPSLOCK: u32 = 58;
        let mut old = KeyState::default_us().unwrap();
        old.update(KEY_CAPSLOCK, true);
        let mut new = KeyState::default_us().unwrap();
        let lock = 1 << new.keymap.mod_get_index(xkb::MOD_NAME_CAPS);
        assert_eq!(new.carry_over(&old, [KEY_CAPSLOCK]).2, lock);
        new.update(KEY_CAPSLOCK, false);
        assert_eq!(new.mods().2, lock);
    }
}
