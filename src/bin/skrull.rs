use std::error::Error;
use std::io::{BufWriter, Read};
use std::io::Write;
use std::path::PathBuf;

use clap::Parser;

use skrull::lexer::Token;
use skrull::transform::java::JavaModule;
use skrull::transform::ts::TsModule;
use skrull::types::Module;

#[derive(clap::Parser, Clone)]
struct ToJavaArgs {
    /// FQDN of the root package to place generated classes into
    root: String,
    /// Output directory where to generate Java files
    out: PathBuf,
}

#[derive(clap::Parser, Clone)]
struct ToTSArgs {
    /// Output file to write TypeScript to (writes to stdout if missing)
    out: Option<PathBuf>,
}

#[derive(clap::Subcommand, Clone)]
enum GenerateAction {
    /// Generate Java code
    Java(ToJavaArgs),
    /// Generate TypeScript code
    TS(ToTSArgs),
}

#[derive(clap::Subcommand, Clone)]
enum Action {
    /// Translate a Skrull module into another supported language
    #[clap(subcommand)]
    Generate(GenerateAction),
}

#[derive(clap::Parser, Clone)]
struct Args {
    #[clap(subcommand)]
    action: Action,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    let mut input = vec![];
    std::io::stdin().read_to_end(&mut input)?;
    let input = String::from_utf8(input)?;
    let parsed = Module::parse_tokens(match args.action {
        Action::Generate(GenerateAction::Java(ToJavaArgs { ref root, .. })) => root,
        Action::Generate(GenerateAction::TS(..)) => "ts"
    }, Token::parse_ascii(input.as_str())?)?;
    Ok(match args.action {
        Action::Generate(GenerateAction::Java(ToJavaArgs { out, .. })) => {
            let jm = JavaModule::try_from(&parsed)?;
            for fqdn in jm.classes() {
                let Some(content) = jm.resolve(fqdn) else { continue; };
                let mut path = out.clone();
                for pkg in fqdn.split('.') {
                    path.push(pkg)
                }
                path.set_extension("java");
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                let mut out = std::fs::File::create(path.as_path())?;
                write!(out, "{}", content)?;
                eprintln!("{}", path.to_string_lossy());
            }
        }
        Action::Generate(GenerateAction::TS(ToTSArgs { out })) => {
            if let Some(path) = out {
                let out = std::fs::File::open(path.as_path())?;
                let mut out = BufWriter::new(out);
                write!(out, "{}", TsModule::try_from(&parsed)?)?;
                eprintln!("{}", path.to_string_lossy());
            } else {
                let out = std::io::stdout();
                let mut out = BufWriter::new(out);
                write!(out, "{}", TsModule::try_from(&parsed)?)?
            }
        }
    })
}
