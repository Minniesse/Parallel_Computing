use ratatui::{
    backend::Backend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Span, Spans},
    widgets::{Axis, Block, Borders, Chart, Dataset, GraphType, Paragraph},
    Frame,
};
use std::collections::HashMap;

use crate::app::App;

pub fn render_performance_view<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),  // Title
            Constraint::Min(10),    // Chart
            Constraint::Length(5),  // Stats
        ])
        .split(area);

    // Title
    let title_block = Block::default()
        .title("Performance Timeline")
        .borders(Borders::ALL);
    f.render_widget(title_block, chunks[0]);

    // Performance chart
    render_performance_chart(f, app, chunks[1]);
    
    // Stats
    render_performance_stats(f, app, chunks[2]);
}

fn render_performance_chart<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    let block = Block::default()
        .title("Execution Time (ms)")
        .borders(Borders::ALL);
    
    if app.performance_history.is_empty() {
        let message = Paragraph::new("No performance data available.")
            .style(Style::default().fg(Color::Gray));
        f.render_widget(block.clone(), area);
        let inner_area = block.inner(area);
        f.render_widget(message, inner_area);
        return;
    }
    
    // Convert timestamps to relative position (0.0 to 1.0) based on time range
    let history = &app.performance_history;
    let first_time = history.first().unwrap().timestamp.timestamp() as f64;
    let last_time = history.last().unwrap().timestamp.timestamp() as f64;
    let time_range = (last_time - first_time).max(1.0); // Avoid division by zero
    
    // Prepare data points for the chart
    let data_points: Vec<(f64, f64)> = history.iter().map(|snapshot| {
        let x = (snapshot.timestamp.timestamp() as f64 - first_time) / time_range;
        let y = snapshot.execution_time * 1000.0; // Convert to milliseconds
        (x, y)
    }).collect();
    
    // Find max value for y-axis scaling
    let max_y = data_points.iter()
        .map(|(_, y)| *y)
        .fold(0.0f64, |max: f64, y| max.max(y))
        .max(0.001f64); // Avoid zero scale
    
    let datasets = vec![
        Dataset::default()
            .name("Execution Time (ms)")
            .marker(symbols::Marker::Braille)
            .style(Style::default().fg(Color::Cyan))
            .graph_type(GraphType::Line)
            .data(&data_points)
    ];
    
    let chart = Chart::new(datasets)
        .block(block)
        .x_axis(
            Axis::default()
                .title("Time")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, 1.0])
                .labels(vec![
                    Span::styled("Oldest", Style::default().fg(Color::Gray)),
                    Span::styled("Latest", Style::default().fg(Color::White)),
                ])
        )
        .y_axis(
            Axis::default()
                .title("ms")
                .style(Style::default().fg(Color::Gray))
                .bounds([0.0, max_y * 1.1]) // Add 10% margin on top
                .labels(vec![
                    Span::styled("0", Style::default().fg(Color::Gray)),
                    Span::styled(format!("{:.1}", max_y / 2.0), Style::default().fg(Color::Gray)),
                    Span::styled(format!("{:.1}", max_y), Style::default().fg(Color::White)),
                ])
        );
    
    f.render_widget(chart, area);
}

fn render_performance_stats<B: Backend>(f: &mut Frame<B>, app: &App, area: Rect) {
    let block = Block::default()
        .title("Performance Statistics")
        .borders(Borders::ALL);
    
    if app.performance_history.is_empty() {
        let message = Paragraph::new("No statistics available.")
            .style(Style::default().fg(Color::Gray));
        f.render_widget(block.clone(), area);
        let inner_area = block.inner(area);
        f.render_widget(message, inner_area);
        return;
    }
    
    // Calculate some basic statistics
    let history = &app.performance_history;
    let exec_times: Vec<f64> = history.iter()
        .map(|snapshot| snapshot.execution_time * 1000.0) // Convert to ms
        .collect();
    
    let avg_time = exec_times.iter().sum::<f64>() / exec_times.len() as f64;
    let min_time = exec_times.iter().fold(f64::INFINITY, |min, &time| min.min(time));
    let max_time = exec_times.iter().fold(0.0f64, |max: f64, &time| max.max(time));
    
    // Get current value
    let current_time = exec_times.last().unwrap_or(&0.0);
    
    let stats_text = vec![
        Spans::from(vec![
            Span::raw("Current: "),
            Span::styled(
                format!("{:.2} ms", current_time),
                Style::default().fg(Color::Cyan)
            ),
            Span::raw("  |  Average: "),
            Span::styled(
                format!("{:.2} ms", avg_time),
                Style::default().fg(Color::Yellow)
            ),
            Span::raw("  |  Min: "),
            Span::styled(
                format!("{:.2} ms", min_time),
                Style::default().fg(Color::Green)
            ),
            Span::raw("  |  Max: "),
            Span::styled(
                format!("{:.2} ms", max_time),
                Style::default().fg(Color::Red)
            ),
        ]),
    ];
    
    let paragraph = Paragraph::new(stats_text)
        .block(block);
    
    f.render_widget(paragraph, area);
}
