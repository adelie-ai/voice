use anyhow::Result;
use tracing_subscriber::EnvFilter;

mod config;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();

    let _config = config::load()?;

    tracing::info!("adele-voice starting");

    // Pipeline wiring will be added in Phase 8.
    // For now, just wait for shutdown signal.
    tokio::signal::ctrl_c().await?;
    tracing::info!("adele-voice shutting down");

    Ok(())
}
