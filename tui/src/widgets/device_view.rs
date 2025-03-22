use ratatui::{
    backend::Backend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Span, Spans},
    widgets::{
        Axis, Block, Borders, Chart, Dataset, GraphType, List, ListItem, 
        Paragraph, Sparkline, Tabs, Wrap,
    },
    Frame,
};
use std::collections::HashMap;

use crate::app::App;
use crate::data::DeviceMetrics;

pub fn render_device_view<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(60),  // Device list
            Constraint::Percentage(40),  // Utilization charts
        ])
        .split(area);

    render_device_list(f, app, chunks[0]);
    render_utilization_charts(f, app, chunks[1]);
}

fn render_device_list<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    let block = Block::default()
        .title("Available Devices")
        .borders(Borders::ALL);
    
    if app.devices.is_empty() {
        let message = Paragraph::new("No devices found.")
            .style(Style::default().fg(Color::Gray));
        f.render_widget(block.clone(), area);
        let inner_area = block.inner(area);
        f.render_widget(message, inner_area);
        return;
    }
    
    let items: Vec<ListItem> = app.devices.iter().enumerate().map(|(i, device)| {
        let memory_percent = (device.memory_used as f64 / device.memory_total as f64 * 100.0).min(100.0);
        let memory_text = format!(
            "{:.1} / {:.1} GB ({:.1}%)",
            device.memory_used as f64 / 1_073_741_824.0,
            device.memory_total as f64 / 1_073_741_824.0,
            memory_percent
        );
        
        let mut spans = vec![
            Span::styled(
                format!("{}: ", device.device_id),
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)
            ),
            Span::styled(
                format!("{} ", device.name),
                Style::default().fg(Color::White)
            ),
            Span::raw(" | Utilization: "),
            Span::styled(
                format!("{:.1}% ", device.utilization),
                Style::default().fg(get_utilization_color(device.utilization))
            ),
            Span::raw(" | Memory: "),
            Span::styled(
                memory_text,
                Style::default().fg(get_utilization_color(memory_percent))
            ),
            Span::raw(" | Operations: "),
            Span::styled(
                format!("{}", device.operations_count),
                Style::default().fg(Color::Yellow)
            ),
        ];
        
        if i == app.selected_device_index {
            spans.insert(0, Span::styled(
                "▶ ",
                Style::default().fg(Color::Yellow)
            ));
        } else {
            spans.insert(0, Span::raw("  "));
        }
        
        ListItem::new(Spans::from(spans))
    }).collect();
    
    let devices = List::new(items)
        .block(block)
        .highlight_style(
            Style::default()
                .bg(Color::DarkGray)
                .add_modifier(Modifier::BOLD)
        )
        .highlight_symbol(">> ");
    
    f.render_widget(devices, area);
}

fn render_utilization_charts<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    let block = Block::default()
        .title("Resource Utilization")
        .borders(Borders::ALL);
    
    if app.resource_history.is_empty() {
        let message = Paragraph::new("No utilization data available.")
            .style(Style::default().fg(Color::Gray));
        f.render_widget(block.clone(), area);
        let inner_area = block.inner(area);
        f.render_widget(message, inner_area);
        return;
    }
    
    let inner_area = block.inner(area);
    f.render_widget(block, area);
    
    let chunks = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(33),  // CPU
            Constraint::Percentage(33),  // Memory
            Constraint::Percentage(34),  // GPU
        ])
        .margin(1)
        .split(inner_area);
    
    // CPU Utilization chart
    let cpu_data: Vec<u64> = app.resource_history.iter()
        .map(|r| r.cpu as u64)
        .collect();
    
    let cpu_sparkline = Sparkline::default()
        .block(Block::default().title("CPU").borders(Borders::ALL))
        .data(&cpu_data)
        .style(Style::default().fg(Color::Cyan))
        .max(100);
    
    f.render_widget(cpu_sparkline, chunks[0]);
    
    // Memory Utilization chart
    let mem_data: Vec<u64> = app.resource_history.iter()
        .map(|r| r.memory as u64)
        .collect();
    
    let mem_sparkline = Sparkline::default()
        .block(Block::default().title("Memory").borders(Borders::ALL))
        .data(&mem_data)
        .style(Style::default().fg(Color::Magenta))
        .max(100);
    
    f.render_widget(mem_sparkline, chunks[1]);
    
    // GPU Utilization chart (use first GPU if available)
    if !app.resource_history.is_empty() && !app.resource_history[0].gpu.is_empty() {
        let gpu_id = app.resource_history[0].gpu.keys().next().unwrap().clone();
        let gpu_data: Vec<u64> = app.resource_history.iter()
            .filter_map(|r| r.gpu.get(&gpu_id).map(|g| g.utilization as u64))
            .collect();
        
        let gpu_sparkline = Sparkline::default()
            .block(Block::default().title(format!("GPU: {}", gpu_id)).borders(Borders::ALL))
            .data(&gpu_data)
            .style(Style::default().fg(Color::Green))
            .max(100);
            
        f.render_widget(gpu_sparkline, chunks[2]);
    } else {
        let gpu_sparkline = Sparkline::default()
            .block(Block::default().title("GPU").borders(Borders::ALL))
            .data(&[0])
            .style(Style::default().fg(Color::Gray))
            .max(100);
            
        f.render_widget(gpu_sparkline, chunks[2]);
    };
}

fn get_utilization_color(utilization: f64) -> Color {
    if utilization < 20.0 {
        Color::Blue
    } else if utilization < 50.0 {
        Color::Green
    } else if utilization < 80.0 {
        Color::Yellow
    } else {
        Color::Red
    }
}
