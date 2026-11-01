//! Exercise the installed binary's entry points without weights or a GPU.
use std::io::Write;
use std::process::{Command, Output, Stdio};

fn run(args: &[&str], input: &str) -> Output {
    run_with_root(args, input, "/dev/null/models")
}

fn run_with_root(args: &[&str], input: &str, root: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_ingot"))
        .args(args)
        // Help must not try to read this deliberately invalid model directory.
        .env("INGOT_MODELS", root)
        .env("HF_ENDPOINT", "http://127.0.0.1:9")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(input.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

#[test]
fn command_help_and_version_do_not_touch_models_or_gpu() {
    for command in [
        "run", "chat", "serve", "generate", "models", "pull", "dev", "service",
    ] {
        for flag in ["--help", "-h"] {
            let out = run(&[command, flag], "");
            assert!(
                out.status.success(),
                "{command}: {}",
                String::from_utf8_lossy(&out.stderr)
            );
            assert!(String::from_utf8_lossy(&out.stdout).contains(command));
        }
    }
    let out = run(&["--version"], "");
    assert!(out.status.success());
    assert_eq!(
        String::from_utf8(out.stdout).unwrap().trim(),
        concat!("ingot ", env!("CARGO_PKG_VERSION"))
    );
}

#[test]
fn no_arguments_in_a_pipe_print_help_instead_of_waiting_for_a_menu() {
    let out = run(&[], "");
    assert!(out.status.success());
    assert!(String::from_utf8_lossy(&out.stdout).contains("ingot run"));
}

#[test]
fn run_requires_model_in_noninteractive_mode_and_rejects_unknown_flags() {
    let out = run(&["run", "--yes"], "");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("specify"));
    let out = run(&["run", "--invalid"], "");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("usage:"));
}

#[test]
fn piped_yes_is_not_download_permission() {
    let root = std::env::temp_dir().join(format!("ingot-cli-no-download-{}", std::process::id()));
    assert!(!root.exists());
    let out = run_with_root(&["run", "qwen3.8-iq2_xs"], "yes\n", root.to_str().unwrap());
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("--yes"));
    assert!(!String::from_utf8_lossy(&out.stderr).contains("Download and start"));
    assert!(!root.exists());
}

#[test]
fn service_install_dry_run_prints_the_agent_without_writing_it() {
    let home = std::env::temp_dir().join(format!("ingot-service-{}", std::process::id()));
    std::fs::create_dir_all(&home).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_ingot"))
        .args(["service", "install", "--dry-run", "--keep-alive", "10m"])
        .env("HOME", &home)
        .env("INGOT_MODELS", "/Volumes/models")
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let plist = home.join("Library/LaunchAgents/com.ingot.serve.plist");
    assert!(
        stdout.contains(&format!("would write {}", plist.display())),
        "{stdout}"
    );
    assert!(stdout.contains(
        "<string>serve</string>\n    <string>--keep-alive</string>\n    <string>10m</string>"
    ));
    assert!(stdout.contains("<string>/Volumes/models</string>"));
    assert!(stdout.contains("launchctl bootstrap gui/"));
    assert!(!home.join("Library").exists());
    let out = run(&["service", "install", "--host", "0.0.0.0"], "");
    assert!(!out.status.success());
    std::fs::remove_dir_all(&home).unwrap();
}

#[test]
fn serve_refuses_a_network_host_without_an_api_key() {
    let out = Command::new(env!("CARGO_BIN_EXE_ingot"))
        .args(["serve", "--host", "0.0.0.0", "--port", "1"])
        .env("INGOT_MODELS", "/dev/null/models")
        .env_remove("INGOT_API_KEY")
        .output()
        .unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--api-key") && stderr.contains("--insecure"),
        "{stderr}"
    );
    let out = run(&["serve", "--thinking", "max"], "");
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("thinking level"));
}
