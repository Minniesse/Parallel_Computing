use ratatui::{
    backend::Backend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Span, Spans, Text},
    widgets::{Block, Borders, Paragraph, Tabs, Widget},
    Frame,
};

use crate::app::{ActiveTab, App, AppState};
use crate::widgets::{
    device_view::render_device_view,
    strategy_view::render_strategy_view,
    performance_view::render_performance_view,
    energy_view::render_energy_view,
    config_editor::render_config_editor,
    help_view::render_help_view,
    comparison_view::render_comparison_view,
};

pub fn draw<B: Backend>(f: &mut Frame<B>, app: &mut App) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Tabs
            Constraint::Min(0),     // Content
            Constraint::Length(3),  // Status bar
        ])
        .split(f.size());

    draw_tabs(f, app, chunks[0]);
    draw_content(f, app, chunks[1]);
    draw_status(f, app, chunks[2]);
}

fn draw_tabs<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    let titles = vec![
        "Devices",
        "Operations",
        "Strategy",
        "Energy",
        "Settings",
    ].iter().map(|t| {
        Spans::from(vec![
            Span::styled(*t, Style::default().fg(Color::White))
        ])
    }).collect();

    let tabs_index = match app.active_tab {
        ActiveTab::Devices => 0,
        ActiveTab::Operations => 1,
        ActiveTab::Strategy => 2, 
        ActiveTab::Energy => 3,
        ActiveTab::Settings => 4,
    };

    let tabs = Tabs::new(titles)
        .block(Block::default().borders(Borders::ALL).title("Parallelism Optimization Dashboard"))
        .select(tabs_index)
        .style(Style::default().fg(Color::Cyan))
        .highlight_style(
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD)
        );

    f.render_widget(tabs, area);
}

fn draw_content<B: Backend>(f: &mut Frame<B>, app: &mut App, area: Rect) {
    match app.state {
        AppState::Normal => {
            match app.active_tab {
                ActiveTab::Devices => render_device_view(f, app, area),
                ActiveTab::Operations => render_operation_view(f, app, area),
                ActiveTab::Strategy => render_strategy_view(f, app, area),
                ActiveTab::Energy => render_energy_view(f, app, area),
                ActiveTab::Settings => render_config_editor(f, app, area),
            }
        },
        AppState::DeviceDetails => render_device_details(f, app, area),
        AppState::StrategyConfiguration => render_strategy_details(f, app, area),
        AppState::PerformanceView => render_performance_view(f, app, area),
        AppState::EnergyMonitor => render_energy_details(f, app, area),
        AppState::ConfigEditor => render_config_editor(f, app, area),
        AppState::Help => render_help_view(f, app, area),
        AppState::Comparison => render_comparison_view(f, app, area),
    }
}

fn render_operation_view<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    let block = Block::default()
        .title("Operation Performance")
        .borders(Borders::ALL);
    f.render_widget(block.clone(), area);
    
    if app.operations.is_empty() {
        let message = Paragraph::new("No operation data available.")
            .style(Style::default().fg(Color::Gray));
        let inner_area = block.inner(area);
        f.render_widget(message, inner_area);
        return;
    }
    
    // Actual operation view implementation would go here
    // This would show each operation's execution time, memory usage, etc.
}

fn render_device_details<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    // If we have a selected device, show its detailed view
    if !app.devices.is_empty() && app.selected_device_index < app.devices.len() {
        let device = &app.devices[app.selected_device_index];
        let block = Block::default()
            .title(format!("Device Details: {}", device.name))
            .borders(Borders::ALL);
        f.render_widget(block, area);
        
        // Detailed device info would go here
    }
}

fn render_strategy_details<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    let block = Block::default()
        .title("Strategy Configuration")
        .borders(Borders::ALL);
    f.render_widget(block, area);
    
    // Strategy configuration details would go here
}

fn render_energy_details<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    let block = Block::default()
        .title("Energy Monitoring Details")
        .borders(Borders::ALL);
    f.render_widget(block, area);
    
    // Detailed energy monitoring view would go here
}

fn draw_status<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(50),
            Constraint::Percentage(50),
        ])
        .split(area);

    // Left side - status and key hints
    let status = match app.state {
        AppState::Normal => "Normal Mode",
        AppState::DeviceDetails => "Device Details",
        AppState::StrategyConfiguration => "Strategy Configuration",
        AppState::PerformanceView => "Performance View",
        AppState::EnergyMonitor => "Energy Monitor",
        AppState::ConfigEditor => "Config Editor",
        AppState::Help => "Help View",
        AppState::Comparison => "Comparison Mode",
    };
    
    let key_hints = match app.state {
        AppState::Normal => "[q] Quit  [h] Help  [d/s/p/e/c] Navigate  [↑/↓/←/→] Select  [Enter] Details",
        AppState::Help => "[q] Back  [↑/↓] Scroll",
        AppState::Comparison => "[q] Back  [r] Run comparison  [s] Save results",
        _ => "[q] Back  [h] Help  [r] Refresh  [↑/↓] Navigate",
    };
    
    let status_text = Paragraph::new(Spans::from(vec![
        Span::styled(status, Style::default().fg(Color::Green)),
        Span::raw(" | "),
        Span::styled(key_hints, Style::default().fg(Color::Gray)),
    ]))
    .block(Block::default().borders(Borders::ALL));
    
    f.render_widget(status_text, chunks[0]);
    
    // Right side - performance metrics
    let performance_text = if let Some(last) = app.performance_history.last() {
        Paragraph::new(Spans::from(vec![
            Span::raw("Execution time: "),
            Span::styled(
                format!("{:.2} ms", last.execution_time * 1000.0),
                Style::default().fg(Color::Yellow)
            ),
            Span::raw(" | Last update: "),
            Span::styled(
                last.timestamp.format("%H:%M:%S").to_string(),
                Style::default().fg(Color::Blue)
            ),
        ]))
    } else {
        Paragraph::new("No performance data available")
    };
    
    let performance_widget = performance_text
        .block(Block::default().borders(Borders::ALL));
    
    f.render_widget(performance_widget, chunks[1]);
}
