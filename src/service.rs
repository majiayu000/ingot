//! `ingot service`: a per-user launchd agent that keeps `ingot serve`
//! running on 127.0.0.1 with logs under `~/Library/Logs/ingot/`.
use crate::{Result, invalid};
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const USAGE: &str = "usage: ingot service install [--dry-run] [--port 8080] [--default MODEL] \
[--keep-alive DURATION] [--context N]
       ingot service uninstall [--dry-run] | status | logs [--lines N]

install writes ~/Library/LaunchAgents/com.ingot.serve.plist, which runs
`ingot serve` without MODEL on 127.0.0.1 at login and restarts it if it
exits with an error, and loads it with launchctl. Logs go to
~/Library/Logs/ingot/. INGOT_MODELS is passed on when it is set.
--dry-run prints the agent and the launchctl commands without running them.";

const LABEL: &str = "com.ingot.serve";
/// `serve` options `install` passes on.
const SERVE_FLAGS: [&str; 4] = ["--port", "--default", "--keep-alive", "--context"];

pub struct Agent {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub models: Option<PathBuf>,
    pub logs: PathBuf,
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

impl Agent {
    pub fn plist(&self) -> String {
        let mut arguments = format!(
            "    <string>{}</string>\n    <string>serve</string>\n",
            escape(&self.program.to_string_lossy())
        );
        for a in &self.args {
            arguments += &format!("    <string>{}</string>\n", escape(a));
        }
        let environment = match &self.models {
            Some(dir) => format!(
                "  <key>EnvironmentVariables</key>\n  <dict>\n    <key>INGOT_MODELS</key>\n    \
                 <string>{}</string>\n  </dict>\n",
                escape(&dir.to_string_lossy())
            ),
            None => String::new(),
        };
        let log = |name: &str| escape(&self.logs.join(name).to_string_lossy());
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{LABEL}</string>
  <key>ProgramArguments</key>
  <array>
{arguments}  </array>
{environment}  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>ProcessType</key>
  <string>Interactive</string>
  <key>StandardOutPath</key>
  <string>{}</string>
  <key>StandardErrorPath</key>
  <string>{}</string>
</dict>
</plist>
"#,
            log("serve.out.log"),
            log("serve.log")
        )
    }
}

struct Paths {
    plist: PathBuf,
    logs: PathBuf,
}
fn paths() -> Result<Paths> {
    let home = std::env::var_os("HOME").ok_or_else(|| invalid("HOME is not set"))?;
    let home = Path::new(&home);
    Ok(Paths {
        plist: home.join(format!("Library/LaunchAgents/{LABEL}.plist")),
        logs: home.join("Library/Logs/ingot"),
    })
}

/// `gui/UID`, the launchd domain of the logged-in user.
fn domain() -> Result<String> {
    let out = Command::new("id").arg("-u").output()?;
    let uid = String::from_utf8_lossy(&out.stdout).trim().to_owned();
    if !out.status.success() || uid.is_empty() {
        return Err(invalid("cannot determine the user id (`id -u` failed)"));
    }
    Ok(format!("gui/{uid}"))
}

fn launchctl(args: &[&str], dry_run: bool) -> Result<bool> {
    println!("launchctl {}", args.join(" "));
    if dry_run {
        return Ok(true);
    }
    Ok(Command::new("launchctl").args(args).status()?.success())
}

/// `--flag value` pairs from `allowed`, plus whether `--dry-run` was given.
fn flags(args: &[OsString], allowed: &[&str]) -> Result<(Vec<String>, bool)> {
    let mut pairs = Vec::new();
    let mut dry_run = false;
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        let arg = arg.to_str().ok_or_else(|| invalid(USAGE))?;
        if arg == "--dry-run" {
            dry_run = true;
            continue;
        }
        if !allowed.contains(&arg) {
            return Err(invalid(format!("unknown option {arg}\n{USAGE}")));
        }
        let value = rest
            .next()
            .and_then(|v| v.to_str())
            .ok_or_else(|| invalid(format!("{arg} requires a value")))?;
        pairs.extend([arg.to_owned(), value.to_owned()]);
    }
    Ok((pairs, dry_run))
}

fn install(args: &[OsString]) -> Result<()> {
    let (serve_args, dry_run) = flags(args, &SERVE_FLAGS)?;
    let paths = paths()?;
    let program = std::env::current_exe()?.canonicalize()?;
    if program.components().any(|c| c.as_os_str() == "target") {
        eprintln!(
            "warning: {} is a build output; install the binary first (scripts/install.sh) \
             so the agent survives `cargo clean`",
            program.display()
        );
    }
    let agent = Agent {
        program,
        args: serve_args,
        models: std::env::var_os("INGOT_MODELS").map(PathBuf::from),
        logs: paths.logs.clone(),
    };
    let plist = agent.plist();
    let domain = domain()?;
    let plist_path = paths.plist.to_string_lossy();
    if dry_run {
        println!("would write {plist_path}:\n{plist}");
        println!("would create {}", paths.logs.display());
    } else {
        fs::create_dir_all(&paths.logs)?;
        if let Some(dir) = paths.plist.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(&paths.plist, &plist)?;
        println!("wrote {plist_path}");
    }
    // Replace a previously loaded agent; failing here only means none was loaded.
    launchctl(&["bootout", &format!("{domain}/{LABEL}")], dry_run)?;
    if !launchctl(&["bootstrap", &domain, &plist_path], dry_run)? {
        return Err(invalid(format!(
            "launchctl bootstrap failed; check {} and `ingot service logs`",
            plist_path
        )));
    }
    let port = agent
        .args
        .iter()
        .position(|a| a == "--port")
        .map_or("8080", |i| agent.args[i + 1].as_str());
    println!(
        "{} ingot serve on http://127.0.0.1:{port}/ (API /v1); `ingot service status` checks it",
        if dry_run { "would start" } else { "started" }
    );
    Ok(())
}

fn uninstall(args: &[OsString]) -> Result<()> {
    let (_, dry_run) = flags(args, &[])?;
    let paths = paths()?;
    launchctl(&["bootout", &format!("{}/{LABEL}", domain()?)], dry_run)?;
    if !paths.plist.exists() {
        println!("{} is not installed", paths.plist.display());
    } else if dry_run {
        println!("would remove {}", paths.plist.display());
    } else {
        fs::remove_file(&paths.plist)?;
        println!("removed {}", paths.plist.display());
    }
    println!("logs are kept in {}", paths.logs.display());
    Ok(())
}

fn status() -> Result<()> {
    let paths = paths()?;
    let installed = paths.plist.exists();
    println!(
        "agent: {} ({})",
        paths.plist.display(),
        if installed {
            "installed"
        } else {
            "not installed"
        }
    );
    let out = Command::new("launchctl")
        .args(["print", &format!("{}/{LABEL}", domain()?)])
        .output()?;
    if !out.status.success() {
        println!("launchd: not loaded");
        return Ok(());
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let field = |key: &str| {
        text.lines()
            .map(str::trim)
            .find_map(|l| l.strip_prefix(key)?.strip_prefix(" = "))
            .unwrap_or("-")
            .to_owned()
    };
    println!(
        "launchd: loaded, state {}, pid {}, last exit {}",
        field("state"),
        field("pid"),
        field("last exit code")
    );
    println!("logs: {}", paths.logs.join("serve.log").display());
    Ok(())
}

fn logs(args: &[OsString]) -> Result<()> {
    let lines: usize = match args {
        [] => 100,
        [flag, n] if flag == "--lines" => n
            .to_str()
            .and_then(|n| n.parse().ok())
            .ok_or_else(|| invalid("--lines must be an integer"))?,
        _ => return Err(invalid(USAGE)),
    };
    let path = paths()?.logs.join("serve.log");
    let text = fs::read_to_string(&path)
        .map_err(|e| std::io::Error::new(e.kind(), format!("{}: {e}", path.display())))?;
    let all: Vec<_> = text.lines().collect();
    for line in &all[all.len().saturating_sub(lines)..] {
        println!("{line}");
    }
    Ok(())
}

pub fn main(args: &[OsString]) -> Result<()> {
    match args.first().and_then(|a| a.to_str()) {
        Some("install") => install(&args[1..]),
        Some("uninstall") => uninstall(&args[1..]),
        Some("status") if args.len() == 1 => status(),
        Some("logs") => logs(&args[1..]),
        None | Some("--help" | "-h" | "help") => {
            println!("{USAGE}");
            Ok(())
        }
        _ => Err(invalid(USAGE)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn plist_runs_serve_with_escaped_arguments_and_logs() {
        let agent = Agent {
            program: "/Users/a b/.local/bin/ingot".into(),
            args: vec!["--default".into(), "x<&>\"".into()],
            models: Some("/Volumes/SSD/models".into()),
            logs: "/Users/a b/Library/Logs/ingot".into(),
        };
        let plist = agent.plist();
        assert!(plist.starts_with("<?xml"));
        assert!(plist.contains("<string>com.ingot.serve</string>"));
        assert!(plist.contains(
            "    <string>/Users/a b/.local/bin/ingot</string>\n    <string>serve</string>\n    \
             <string>--default</string>\n    <string>x&lt;&amp;&gt;&quot;</string>\n  </array>"
        ));
        assert!(
            plist.contains("<key>INGOT_MODELS</key>\n    <string>/Volumes/SSD/models</string>")
        );
        assert!(plist.contains("<string>/Users/a b/Library/Logs/ingot/serve.log</string>"));
        assert!(plist.contains("<key>SuccessfulExit</key>\n    <false/>"));
        // Every opened element is closed.
        for tag in ["dict", "array", "plist"] {
            assert_eq!(
                plist.matches(&format!("<{tag}")).count(),
                plist.matches(&format!("</{tag}>")).count(),
                "{tag}"
            );
        }
        let plain = Agent {
            models: None,
            ..agent
        }
        .plist();
        assert!(!plain.contains("EnvironmentVariables"));
    }
    #[test]
    fn install_flags_pass_serve_options_and_reject_others() {
        let args = |a: &[&str]| a.iter().map(OsString::from).collect::<Vec<_>>();
        let (pairs, dry) = flags(
            &args(&["--dry-run", "--port", "9", "--keep-alive", "5m"]),
            &SERVE_FLAGS,
        )
        .unwrap();
        assert!(dry);
        assert_eq!(pairs, ["--port", "9", "--keep-alive", "5m"]);
        assert!(flags(&args(&["--host", "0.0.0.0"]), &SERVE_FLAGS).is_err());
        assert!(flags(&args(&["--port"]), &SERVE_FLAGS).is_err());
    }
}
