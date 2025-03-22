use ratatui::{
    backend::Backend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    text::{Span, Spans},
    widgets::{Block, Borders, Paragraph, Wrap},
    Frame,
};

use crate::app::App;

pub fn render_help_view<B: Backend>(f: &mut Frame<B>, _app: &App, area: Rect) {
    let block = Block::default()
        .title("Help")
        .borders(Borders::ALL);
    
    let help_text = vec![
        Spans::from(Span::styled(
            "Parallelism Optimization Framework - Terminal UI",
            Style::default().fg(Color::Cyan)
        )),
        Spans::from(""),
        Spans::from(Span::styled(
            "Navigation Keys:",
            Style::default().fg(Color::Yellow)
        )),
        Spans::from("  q - Quit / Go back to main view"),
        Spans::from("  h - Toggle this help page"),
        Spans::from("  d - Go to Devices tab"),
        Spans::from("  s - Go to Strategy tab"),
        Spans::from("  p - Toggle Performance view"),
        Spans::from("  e - Go to Energy tab"),
        Spans::from("  c - Go to Configuration tab"),
        Spans::from("  r - Refresh data from server"),
        Spans::from("  m - View Performance Comparison"),
        Spans::from(""),
        Spans::from(Span::styled(
            "Selection Controls:",
            Style::default().fg(Color::Yellow)
        )),
        Spans::from("  ←/→ - Navigate between tabs"),
        Spans::from("  ↑/↓ - Select items in current view"),
        Spans::from("  Enter - View details or toggle options"),
        Spans::from(""),
        Spans::from(Span::styled(
            "Tabs:",
            Style::default().fg(Color::Yellow)
        )),
        Spans::from("  Devices - Monitor hardware utilization"),
        Spans::from("  Operations - View operation performance"),
        Spans::from("  Strategy - See current distribution strategy"),
        Spans::from("  Energy - Monitor energy consumption"),
        Spans::from("  Settings - Configure framework parameters"),
        Spans::from("  Comparison - Compare baseline vs. optimized performance"),
        Spans::from(""),
        Spans::from(Span::styled(
            "Tips:",
            Style::default().fg(Color::Yellow)
        )),
        Spans::from("  • Press Enter on devices to see detailed metrics"),
        Spans::from("  • Use the Performance view (p) to monitor execution time trends"),
        Spans::from("  • Toggle energy-aware optimization in the Settings tab"),
        Spans::from("  • Run comparisons with 'r' in the Comparison view"),
        Spans::from("  • Save comparison results with 's' in the Comparison view"),
        Spans::from(""),
        Spans::from(Span::styled(
            "Configuration Options:",
            Style::default().fg(Color::Yellow)
        )),
        Spans::from("  energy_aware - Optimize for energy efficiency"),
        Spans::from("  communication_aware - Optimize data movement between devices"),
        Spans::from("  enable_monitoring - Collect and display performance metrics"),
        Spans::from("  dynamic_adjustment - Adjust strategy during execution"),
        Spans::from("  memory_fraction - Maximum fraction of memory to use (0.0-1.0)"),
        Spans::from(""),
        Spans::from(Span::styled(
            "Performance Monitoring:",
            Style::default().fg(Color::Yellow)
        )),
        Spans::from("  The Performance view shows execution time trends"),
        Spans::from("  You can see device utilization in the Devices tab"),
        Spans::from("  The Energy tab shows power consumption by device"),
        Spans::from("  Strategy view shows the current workload distribution"),
        Spans::from(""),
        Spans::from(Span::styled(
            "Press 'q' to exit this help page",
            Style::default().fg(Color::Green)
        )),
    ];
    
    let help_paragraph = Paragraph::new(help_text)
        .block(block)
        .wrap(Wrap { trim: true });
    
    f.render_widget(help_paragraph, area);
}
