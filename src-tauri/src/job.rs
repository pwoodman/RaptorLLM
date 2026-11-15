//! Keeps llama-server from outliving the app.
//!
//! On Windows every llama-server process is assigned to one Job Object created with
//! `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`. The job handle belongs to the app process, so when
//! the app exits for any reason, including a hard kill or a crash, the OS closes the handle
//! and terminates the child. A spawn that cannot be placed in the job is stopped and reported
//! as an error; it is never left running unprotected.
//!
//! The real spawn is delegated to [`SystemSpawner`]. The only window without protection is
//! between the child starting and the assignment; llama-server does not start helper
//! processes in that window.

use async_trait::async_trait;
use raptor_core::Result;
use raptor_runtime::process::{ProcessHandle, SpawnSpec, Spawner, SystemSpawner};
use std::time::Duration;

/// Spawns llama-server with [`SystemSpawner`] and places it in the app's kill-on-close job.
#[derive(Debug, Default, Clone, Copy)]
pub struct JobSpawner;

#[async_trait]
impl Spawner for JobSpawner {
    async fn spawn(&self, spec: &SpawnSpec) -> Result<ProcessHandle> {
        let handle = SystemSpawner.spawn(spec).await?;
        let Some(pid) = handle.control.pid() else {
            // Without a pid the child cannot be placed in the job, so it is stopped.
            let _ = handle.control.request_stop(Duration::ZERO).await;
            return Err(raptor_core::RaptorError::Runtime(
                "llama-server started without a process id; refusing to run it unprotected".into(),
            ));
        };
        if let Err(reason) = platform::assign(pid) {
            let _ = handle.control.request_stop(Duration::ZERO).await;
            return Err(raptor_core::RaptorError::Runtime(format!(
                "could not place llama-server in a job object ({reason}); it was stopped"
            )));
        }
        Ok(handle)
    }
}

#[cfg(windows)]
mod platform {
    use std::sync::Mutex;
    use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
        SetInformationJobObject, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::Threading::{
        OpenProcess, PROCESS_SET_QUOTA, PROCESS_TERMINATE,
    };

    /// The job handle as an integer, so it can live in a static. Created on first use and
    /// never closed while the app runs; the OS closes it at process exit.
    static JOB: Mutex<Option<usize>> = Mutex::new(None);

    fn job() -> std::result::Result<HANDLE, String> {
        let mut slot = JOB.lock().unwrap_or_else(|p| p.into_inner());
        if let Some(h) = *slot {
            return Ok(h as HANDLE);
        }
        // SAFETY: null attributes and name are valid for CreateJobObjectW.
        let h = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if h.is_null() {
            return Err(format!("CreateJobObjectW failed (error {})", last_error()));
        }
        // SAFETY: the struct is plain data; a zeroed value is a valid starting point.
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        // SAFETY: `info` is a valid JOBOBJECT_EXTENDED_LIMIT_INFORMATION of the stated size.
        let ok = unsafe {
            SetInformationJobObject(
                h,
                JobObjectExtendedLimitInformation,
                (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            )
        };
        if ok == 0 {
            let err = last_error();
            // SAFETY: `h` was returned by CreateJobObjectW and is not used again.
            unsafe { CloseHandle(h) };
            return Err(format!("SetInformationJobObject failed (error {err})"));
        }
        *slot = Some(h as usize);
        Ok(h)
    }

    /// Assigns the process `pid` to the app's job.
    pub fn assign(pid: u32) -> std::result::Result<(), String> {
        let job = job()?;
        // SAFETY: plain Win32 call; the process handle is closed below in every path.
        let proc = unsafe { OpenProcess(PROCESS_SET_QUOTA | PROCESS_TERMINATE, 0, pid) };
        if proc.is_null() {
            return Err(format!("OpenProcess failed (error {})", last_error()));
        }
        // SAFETY: both handles are valid for the duration of the call.
        let ok = unsafe { AssignProcessToJobObject(job, proc) };
        let err = if ok == 0 { last_error() } else { 0 };
        // SAFETY: `proc` came from OpenProcess and is closed exactly once.
        unsafe { CloseHandle(proc) };
        if ok == 0 {
            return Err(format!("AssignProcessToJobObject failed (error {err})"));
        }
        Ok(())
    }

    fn last_error() -> u32 {
        // SAFETY: GetLastError has no preconditions.
        unsafe { windows_sys::Win32::Foundation::GetLastError() }
    }

    #[cfg(test)]
    pub fn job_handle_for_tests() -> std::result::Result<HANDLE, String> {
        job()
    }
}

#[cfg(not(windows))]
mod platform {
    /// Non-Windows builds rely on the process group kill in the process layer.
    pub fn assign(_pid: u32) -> std::result::Result<(), String> {
        Ok(())
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use windows_sys::Win32::Foundation::CloseHandle;
    use windows_sys::Win32::System::JobObjects::{
        IsProcessInJob, JobObjectExtendedLimitInformation, QueryInformationJobObject,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::Threading::{OpenProcess, PROCESS_QUERY_LIMITED_INFORMATION};

    /// A long-running child that exits on its own after the test, if not stopped earlier.
    fn sleeper() -> SpawnSpec {
        SpawnSpec {
            program: "C:/Windows/System32/ping.exe".into(),
            args: vec!["-n".into(), "60".into(), "127.0.0.1".into()],
            env: Vec::new(),
        }
    }

    #[tokio::test]
    async fn spawned_child_is_inside_a_kill_on_close_job() {
        let handle = JobSpawner.spawn(&sleeper()).await.unwrap();
        let pid = handle.control.pid().expect("pid");
        let job = platform::job_handle_for_tests().unwrap();

        // SAFETY: plain Win32 queries on handles this test owns and closes.
        unsafe {
            let proc = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid);
            assert!(!proc.is_null());
            let mut inside = 0;
            assert_ne!(IsProcessInJob(proc, job, &mut inside), 0);
            assert_ne!(inside, 0, "child must be assigned to the app job");

            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            let ok = QueryInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                (&mut info as *mut JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                std::ptr::null_mut(),
            );
            assert_ne!(ok, 0);
            assert_ne!(
                info.BasicLimitInformation.LimitFlags & JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
                0
            );
            CloseHandle(proc);
        }
        handle
            .control
            .request_stop(Duration::from_millis(200))
            .await
            .unwrap();
    }
}
