use ratatui::{
    backend::Backend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Span, Spans},
    widgets::{Axis, BarChart, Block, Borders, Chart, Dataset, GraphType, Paragraph},
    Frame,
};
use std::collections::HashMap;
use itertools::Itertools;

use crate::app::App;
use crate::data::DeviceEnergy;

pub fn render_energy_view<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(5),  // Summary
            Constraint::Min(10),    // Charts
        ])
        .split(area);

    render_energy_summary(f, app, chunks[0]);
    render_energy_charts(f, app, chunks[1]);
}

fn render_energy_summary<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    let block = Block::default()
        .title("Energy Consumption Summary")
        .borders(Borders::ALL);
    
    let energy = &app.energy;
    
    let last_update = energy.timestamp.format("%Y-%m-%d %H:%M:%S").to_string();
    
    let text = vec![
        Spans::from(vec![
            Span::raw("Total Energy: "),
            Span::styled(
                format!("{:.2} joules ({:.3} Wh)", energy.total_energy_joules, energy.total_energy_watt_hours),
                Style::default().fg(Color::Green)
            ),
            Span::raw("  |  Last Update: "),
            Span::styled(
                last_update,
                Style::default().fg(Color::Blue)
            ),
        ]),
        Spans::from(vec![
            Span::styled(
                "Press Enter for detailed energy monitoring view",
                Style::default().fg(Color::Gray)
            ),
        ]),
    ];
    
    let paragraph = Paragraph::new(text)
        .block(block);
    
    f.render_widget(paragraph, area);
}

fn render_energy_charts<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    let block = Block::default()
        .title("Device Energy Consumption")
        .borders(Borders::ALL);
    
    let energy = &app.energy;
    
    if energy.device_energy.is_empty() {
        let message = Paragraph::new("No energy data available.")
            .style(Style::default().fg(Color::Gray));
        f.render_widget(block.clone(), area);
        let inner_area = block.inner(area);
        f.render_widget(message, inner_area);
        return;
    }
    
    let inner_area = block.inner(area);
    f.render_widget(block, area);
    
    // Sort devices by energy consumption (descending)
    let mut device_data: Vec<(&String, &DeviceEnergy)> = energy.device_energy.iter().collect();
    device_data.sort_by(|a, b| b.1.energy_joules.partial_cmp(&a.1.energy_joules).unwrap());
    
    // Limit to top 5 devices for clarity
    let top_devices: Vec<(&String, &DeviceEnergy)> = device_data.into_iter().take(5).collect();
    
    // Create bar chart data
    let device_names: Vec<String> = top_devices.iter().map(|(name, _)| (*name).clone()).collect();
    
    let energy_values: Vec<u64> = top_devices.iter()
        .map(|(_, energy)| (energy.energy_joules * 100.0) as u64) // Scale for better visualization
        .collect();
    
    // Create pairs with correct types for BarChart
    let chart_data: Vec<(&str, u64)> = device_names.iter()
        .map(|s| s.as_str())
        .zip(energy_values.iter().copied())
        .collect();
    
    let bar_chart = BarChart::default()
        .block(Block::default())
        .data(&chart_data)
        .bar_width(9)
        .bar_gap(3)
        .bar_style(Style::default().fg(Color::Green))
        .value_style(Style::default().fg(Color::White).add_modifier(Modifier::BOLD));
    
    f.render_widget(bar_chart, inner_area);
}
