use std::collections::HashMap;
use anyhow::Result;
use tui_logger::TuiLoggerWidget;

use crate::communication::PythonFrameworkClient;
use crate::data::{
    DeviceMetrics, EnergyMetrics, OperationMetrics, 
    PerformanceSnapshot, StrategyConfig, ResourceUtilization,
    ComparisonResult,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AppState {
    Normal,
    DeviceDetails,
    StrategyConfiguration,
    PerformanceView,
    EnergyMonitor,
    ConfigEditor,
    Help,
    Comparison,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ActiveTab {
    Devices,
    Operations,
    Strategy,
    Energy,
    Settings,
}

pub struct App {
    pub state: AppState,
    pub active_tab: ActiveTab,
    pub should_quit: bool,
    pub framework_client: PythonFrameworkClient,
    
    // Data
    pub devices: Vec<DeviceMetrics>,
    pub operations: Vec<OperationMetrics>,
    pub strategy: StrategyConfig,
    pub energy: EnergyMetrics,
    pub resource_history: Vec<ResourceUtilization>,
    
    // UI state
    pub selected_device_index: usize,
    pub selected_operation_index: usize,
    pub selected_config_option: usize,
    pub show_help: bool,
    pub config_options: Vec<(String, String)>,
    
    // Performance history
    pub performance_history: Vec<PerformanceSnapshot>,
    pub history_max_size: usize,

    // Comparison
    pub comparison_result: ComparisonResult,
    pub is_comparison_running: bool,
}

impl App {
    pub fn new(client: PythonFrameworkClient) -> Self {
        let mut app = Self {
            state: AppState::Normal,
            active_tab: ActiveTab::Devices,
            should_quit: false,
            framework_client: client,
            
            devices: Vec::new(),
            operations: Vec::new(),
            strategy: StrategyConfig::default(),
            energy: EnergyMetrics::default(),
            resource_history: Vec::new(),
            
            selected_device_index: 0,
            selected_operation_index: 0,
            selected_config_option: 0,
            show_help: false,
            config_options: vec![
                ("energy_aware".to_string(), "true".to_string()),
                ("communication_aware".to_string(), "true".to_string()),
                ("enable_monitoring".to_string(), "true".to_string()),
                ("dynamic_adjustment".to_string(), "true".to_string()),
                ("memory_fraction".to_string(), "0.9".to_string()),
            ],
            
            performance_history: Vec::new(),
            history_max_size: 100,

            comparison_result: ComparisonResult::default(),
            is_comparison_running: false,
        };
        
        // Load initial data
        let _ = app.refresh_data();
        
        app
    }
    
    pub fn on_tick(&mut self) {
        // Update real-time data at regular intervals
        if let Ok(metrics) = self.framework_client.get_current_metrics() {
            // Update resource utilization history
            if self.resource_history.len() >= self.history_max_size {
                self.resource_history.remove(0);
            }
            self.resource_history.push(metrics.resource_utilization);
            
            // Update performance history
            if self.performance_history.len() >= self.history_max_size {
                self.performance_history.remove(0);
            }
            self.performance_history.push(PerformanceSnapshot {
                timestamp: chrono::Utc::now(),
                execution_time: metrics.avg_execution_time,
                device_utilization: metrics.device_utilization.clone(),
            });
        }

        // Check comparison status if a comparison is running
        if self.is_comparison_running {
            let _ = self.check_comparison_status();
        }
    }
    
    pub fn refresh_data(&mut self) -> Result<()> {
        // Fetch current metrics
        if let Ok(metrics) = self.framework_client.get_current_metrics() {
            // Update device metrics
            self.devices = metrics.devices;
            
            // Update operation metrics
            self.operations = metrics.operations;
            
            // Update device utilization
            if !self.resource_history.is_empty() {
                self.resource_history.clear();
            }
            self.resource_history.push(metrics.resource_utilization);
        }
        
        // Fetch energy metrics
        if let Ok(energy_data) = self.framework_client.get_energy_metrics() {
            self.energy = energy_data;
        }
        
        // Fetch current strategy
        if let Ok(strategy) = self.framework_client.get_current_strategy() {
            self.strategy = strategy;
        }
        
        Ok(())
    }
    
    // Navigation handlers
    pub fn on_key_h(&mut self) {
        self.state = if self.state == AppState::Help {
            AppState::Normal
        } else {
            AppState::Help
        };
    }
    
    pub fn on_key_d(&mut self) {
        if self.state == AppState::Normal {
            self.active_tab = ActiveTab::Devices;
        } else if self.state != AppState::DeviceDetails {
            self.state = AppState::Normal;
            self.active_tab = ActiveTab::Devices;
        } else {
            self.state = AppState::Normal;
        }
    }
    
    pub fn on_key_s(&mut self) {
        if self.state == AppState::Normal {
            self.active_tab = ActiveTab::Strategy;
        } else if self.state != AppState::StrategyConfiguration {
            self.state = AppState::Normal;
            self.active_tab = ActiveTab::Strategy;
        } else {
            self.state = AppState::Normal;
        }
    }
    
    pub fn on_key_p(&mut self) {
        self.state = if self.state == AppState::PerformanceView {
            AppState::Normal
        } else {
            AppState::PerformanceView
        };
    }
    
    pub fn on_key_e(&mut self) {
        if self.state == AppState::Normal {
            self.active_tab = ActiveTab::Energy;
        } else if self.state != AppState::EnergyMonitor {
            self.state = AppState::Normal;
            self.active_tab = ActiveTab::Energy;
        } else {
            self.state = AppState::Normal;
        }
    }
    
    pub fn on_key_c(&mut self) {
        if self.state == AppState::Normal {
            self.active_tab = ActiveTab::Settings;
        } else if self.state != AppState::ConfigEditor {
            self.state = AppState::Normal;
            self.active_tab = ActiveTab::Settings;
        } else {
            self.state = AppState::Normal;
        }
    }
    
    pub fn on_up(&mut self) {
        match self.state {
            AppState::Normal => {
                match self.active_tab {
                    ActiveTab::Devices => {
                        if !self.devices.is_empty() {
                            self.selected_device_index = self.selected_device_index
                                .checked_sub(1)
                                .unwrap_or(self.devices.len() - 1);
                        }
                    }
                    ActiveTab::Operations => {
                        if !self.operations.is_empty() {
                            self.selected_operation_index = self.selected_operation_index
                                .checked_sub(1)
                                .unwrap_or(self.operations.len() - 1);
                        }
                    }
                    ActiveTab::Settings => {
                        if !self.config_options.is_empty() {
                            self.selected_config_option = self.selected_config_option
                                .checked_sub(1)
                                .unwrap_or(self.config_options.len() - 1);
                        }
                    }
                    _ => {}
                }
            }
            AppState::ConfigEditor => {
                if !self.config_options.is_empty() {
                    self.selected_config_option = self.selected_config_option
                        .checked_sub(1)
                        .unwrap_or(self.config_options.len() - 1);
                }
            }
            _ => {}
        }
    }
    
    pub fn on_down(&mut self) {
        match self.state {
            AppState::Normal => {
                match self.active_tab {
                    ActiveTab::Devices => {
                        if !self.devices.is_empty() {
                            self.selected_device_index = (self.selected_device_index + 1) % self.devices.len();
                        }
                    }
                    ActiveTab::Operations => {
                        if !self.operations.is_empty() {
                            self.selected_operation_index = (self.selected_operation_index + 1) % self.operations.len();
                        }
                    }
                    ActiveTab::Settings => {
                        if !self.config_options.is_empty() {
                            self.selected_config_option = (self.selected_config_option + 1) % self.config_options.len();
                        }
                    }
                    _ => {}
                }
            }
            AppState::ConfigEditor => {
                if !self.config_options.is_empty() {
                    self.selected_config_option = (self.selected_config_option + 1) % self.config_options.len();
                }
            }
            _ => {}
        }
    }
    
    pub fn on_left(&mut self) {
        match self.state {
            AppState::Normal => {
                // Cycle tabs backward
                self.active_tab = match self.active_tab {
                    ActiveTab::Devices => ActiveTab::Settings,
                    ActiveTab::Operations => ActiveTab::Devices,
                    ActiveTab::Strategy => ActiveTab::Operations,
                    ActiveTab::Energy => ActiveTab::Strategy,
                    ActiveTab::Settings => ActiveTab::Energy,
                };
            }
            _ => {}
        }
    }
    
    pub fn on_right(&mut self) {
        match self.state {
            AppState::Normal => {
                // Cycle tabs forward
                self.active_tab = match self.active_tab {
                    ActiveTab::Devices => ActiveTab::Operations,
                    ActiveTab::Operations => ActiveTab::Strategy,
                    ActiveTab::Strategy => ActiveTab::Energy,
                    ActiveTab::Energy => ActiveTab::Settings,
                    ActiveTab::Settings => ActiveTab::Devices,
                };
            }
            _ => {}
        }
    }
    
    pub fn on_enter(&mut self) {
        match self.state {
            AppState::Normal => {
                match self.active_tab {
                    ActiveTab::Devices => {
                        if !self.devices.is_empty() {
                            self.state = AppState::DeviceDetails;
                        }
                    }
                    ActiveTab::Strategy => {
                        self.state = AppState::StrategyConfiguration;
                    }
                    ActiveTab::Energy => {
                        self.state = AppState::EnergyMonitor;
                    }
                    ActiveTab::Settings => {
                        self.state = AppState::ConfigEditor;
                    }
                    _ => {}
                }
            }
            AppState::ConfigEditor => {
                // Toggle boolean values or handle editing
                if !self.config_options.is_empty() {
                    let (key, value) = &mut self.config_options[self.selected_config_option];
                    if value == "true" {
                        *value = "false".to_string();
                    } else if value == "false" {
                        *value = "true".to_string();
                    }
                    // For non-boolean values, we would enter an edit mode
                    
                    // Apply the configuration change
                    let _ = self.framework_client.update_config(key, value);
                }
            }
            _ => {}
        }
    }

    pub fn run_comparison(&mut self) -> Result<()> {
        self.is_comparison_running = true;
        match self.framework_client.run_comparison() {
            Ok(result) => {
                self.comparison_result = result;
                Ok(())
            },
            Err(e) => {
                self.is_comparison_running = false;
                Err(e)
            }
        }
    }
    
    pub fn check_comparison_status(&mut self) -> Result<()> {
        if self.is_comparison_running {
            match self.framework_client.get_comparison_status() {
                Ok(result) => {
                    self.comparison_result = result.clone();
                    if result.completed {
                        self.is_comparison_running = false;
                    }
                    Ok(())
                },
                Err(e) => {
                    self.is_comparison_running = false;
                    Err(e)
                }
            }
        } else {
            Ok(())
        }
    }
    
    pub fn on_key_compare(&mut self) {
        self.state = AppState::Comparison;
    }
}
