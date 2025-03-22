use anyhow::{Context, Result};
use crossterm::{
    event::{self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::{backend::CrosstermBackend, Terminal};
use std::{io, time::{Duration, Instant}};

mod app;
mod ui;
mod widgets;
mod data;
mod communication;

use app::{App, AppState};
use communication::PythonFrameworkClient;

fn main() -> Result<()> {
    // Initialize terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Create app state
    let client = PythonFrameworkClient::new("http://localhost:8000")?; // Connect to Python API
    let app = App::new(client);
    
    // Run the main loop
    let res = run_app(&mut terminal, app);

    // Restore terminal
    disable_raw_mode()?;
    execute!(
        terminal.backend_mut(),
        LeaveAlternateScreen,
        DisableMouseCapture
    )?;
    terminal.show_cursor()?;

    if let Err(err) = res {
        println!("{err:?}");
    }

    Ok(())
}

fn run_app<B: ratatui::backend::Backend>(
    terminal: &mut Terminal<B>,
    mut app: App,
) -> Result<()> {
    let tick_rate = Duration::from_millis(250);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| ui::draw(f, &mut app))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let Event::Key(key) = event::read()? {
                if key.kind == KeyEventKind::Press {
                    match key.code {
                        KeyCode::Char('q') => {
                            if app.state == AppState::Normal {
                                return Ok(());
                            } else {
                                app.state = AppState::Normal;
                            }
                        },
                        KeyCode::Char('h') => app.on_key_h(),
                        KeyCode::Char('d') => app.on_key_d(),
                        KeyCode::Char('s') => {
                            if app.state == AppState::Comparison && !app.is_comparison_running {
                                app.framework_client.save_comparison_results("comparison_results.json")?;
                            } else {
                                app.on_key_s();
                            }
                        },
                        KeyCode::Char('p') => app.on_key_p(),
                        KeyCode::Char('e') => app.on_key_e(),
                        KeyCode::Char('c') => app.on_key_c(),
                        KeyCode::Char('r') => {
                            if app.state == AppState::Comparison {
                                app.run_comparison()?;
                            } else {
                                app.refresh_data()?;
                            }
                        },
                        KeyCode::Char('m') => app.on_key_compare(),
                        KeyCode::Left => app.on_left(),
                        KeyCode::Right => app.on_right(),
                        KeyCode::Up => app.on_up(),
                        KeyCode::Down => app.on_down(),
                        KeyCode::Enter => app.on_enter(),
                        _ => {}
                    }
                }
            }
        }

        if last_tick.elapsed() >= tick_rate {
            app.on_tick();
            last_tick = Instant::now();
        }
    }
}
