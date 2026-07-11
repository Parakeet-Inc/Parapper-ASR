//! System Audio Recording permission for macOS `CoreAudio` loopback capture.
//!
//! A `CoreAudio` process tap requires the "System Audio Recording" permission
//! (`kTCCServiceAudioCapture`), which is distinct from the microphone permission.
//! Without it the tap is created successfully but records silence — macOS neither
//! returns an error nor shows a prompt on its own. We therefore check and, when
//! needed, explicitly request the permission via the private TCC framework.
//!
//! This mirrors the approach in RustAudio/cpal#1257 (not yet released in the cpal
//! version we depend on). On non-macOS platforms these are no-ops.

use anyhow::{Result, bail};

/// The outcome of ensuring the process may capture system audio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(target_os = "macos"), allow(dead_code))]
pub enum SystemAudioPermission {
    /// Permission is granted; loopback capture will receive real audio.
    Granted,
    /// Permission was denied (either just now or previously). The user must grant
    /// it in System Settings → Privacy & Security → System Audio Recording.
    Denied,
}

impl SystemAudioPermission {
    pub fn is_granted(self) -> bool {
        matches!(self, Self::Granted)
    }
}

/// Returns an error unless the process is allowed to capture system audio.
///
/// If permission has not been granted yet, this shows the system prompt and
/// blocks until the user responds. Once granted, subsequent calls return
/// immediately.
pub(crate) fn ensure_system_audio_permission() -> Result<()> {
    match request_system_audio_permission() {
        SystemAudioPermission::Granted => Ok(()),
        SystemAudioPermission::Denied => bail!(
            "System Audio Recording permission is required to capture speaker/loopback audio. \
             Enable it for Parapper in System Settings → Privacy & Security → System Audio Recording."
        ),
    }
}

#[cfg(target_os = "macos")]
mod imp {
    use std::{ffi::c_void, ptr::from_ref, sync::OnceLock};

    use block2::StackBlock;
    use libloading::{Library, Symbol};
    use objc2_core_foundation::{CFRetained, CFString};

    const TCC_FRAMEWORK: &str = "/System/Library/PrivateFrameworks/TCC.framework/Versions/A/TCC";
    const TCC_SERVICE: &str = "kTCCServiceAudioCapture";

    fn load_tcc() -> Option<&'static Library> {
        static LIBRARY: OnceLock<Option<Library>> = OnceLock::new();
        LIBRARY
            .get_or_init(|| unsafe { Library::new(TCC_FRAMEWORK) }.ok())
            .as_ref()
    }

    fn tcc_service() -> CFRetained<CFString> {
        CFString::from_str(TCC_SERVICE)
    }

    /// Non-blocking check; never shows UI. `false` if denied, undetermined, or
    /// unavailable.
    pub(super) fn check() -> bool {
        let Some(lib) = load_tcc() else {
            return false;
        };
        // SAFETY: `TCCAccessPreflight(CFStringRef, CFDictionaryRef) -> i32` is the
        // documented-by-reverse-engineering signature; we pass a valid service
        // string and a null options dictionary and only read the returned status.
        unsafe {
            let Ok(preflight): Result<
                Symbol<unsafe extern "C" fn(*const c_void, *const c_void) -> i32>,
                _,
            > = lib.get(b"TCCAccessPreflight\0") else {
                return false;
            };
            let service = tcc_service();
            preflight(from_ref(&*service).cast::<c_void>(), std::ptr::null()) == 0
        }
    }

    /// Requests the permission, showing the system prompt if the decision has not
    /// been made yet. Blocks until the user responds. Returns `false` immediately
    /// (no UI) if it was previously denied.
    pub(super) fn request() -> bool {
        if check() {
            return true;
        }
        let Some(lib) = load_tcc() else {
            return false;
        };
        // SAFETY: `TCCAccessRequest(CFStringRef, CFDictionaryRef, block)` invokes
        // the completion block with a `BOOL` result. We keep the sender alive via a
        // raw pointer stored as `usize` so TCC's internal block copy cannot
        // double-drop it, and reclaim it exactly once inside the block.
        unsafe {
            let Ok(request): Result<
                Symbol<unsafe extern "C" fn(*const c_void, *const c_void, *const c_void)>,
                _,
            > = lib.get(b"TCCAccessRequest\0") else {
                return false;
            };

            let (tx, rx) = std::sync::mpsc::sync_channel::<bool>(1);
            let tx_ptr = Box::into_raw(Box::new(tx)) as usize;

            let completion = StackBlock::new(move |granted: u8| {
                let tx = Box::from_raw(tx_ptr as *mut std::sync::mpsc::SyncSender<bool>);
                tx.send(granted != 0).ok();
            });

            let service = tcc_service();
            request(
                from_ref(&*service).cast::<c_void>(),
                std::ptr::null(),
                from_ref(&completion).cast::<c_void>(),
            );

            rx.recv().unwrap_or(false)
        }
    }

    /// Opens Privacy & Security → System Audio Recording in System Settings.
    pub(super) fn open_settings() {
        std::process::Command::new("open")
            .arg(
                "x-apple.systempreferences:com.apple.settings.PrivacySecurity.extension\
                 ?Privacy_AudioCapture",
            )
            .spawn()
            .ok();
    }
}

/// Requests system audio recording permission, showing the prompt if needed.
/// Returns immediately as [`SystemAudioPermission::Granted`] when it was already
/// granted; otherwise blocks until the user responds to the prompt.
#[cfg(target_os = "macos")]
pub fn request_system_audio_permission() -> SystemAudioPermission {
    if imp::request() {
        SystemAudioPermission::Granted
    } else {
        SystemAudioPermission::Denied
    }
}

/// Opens the System Settings pane where the permission can be granted manually.
#[cfg(target_os = "macos")]
pub fn open_system_audio_settings() {
    imp::open_settings();
}

// Non-macOS platforms reach loopback only through hosts that do not need this
// permission (e.g. WASAPI), so these are no-ops that report success.
#[cfg(not(target_os = "macos"))]
pub fn request_system_audio_permission() -> SystemAudioPermission {
    SystemAudioPermission::Granted
}

#[cfg(not(target_os = "macos"))]
pub fn open_system_audio_settings() {}
