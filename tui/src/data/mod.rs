use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

pub mod metrics;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeviceMetrics {
    pub device_id: String,
    pub name: String,
    pub utilization: f64,
    pub memory_used: u64,
    pub memory_total: u64,
    pub operations_count: usize,
    pub compute_time: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OperationMetrics {
    pub op_id: String,
    pub device_id: String,
    pub execution_time: f64,
    pub memory_usage: u64,
    pub flops: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeviceAssignment {
    pub device_id: String,
    pub operation_ids: Vec<String>,
    pub estimated_compute_time: f64,
    pub estimated_memory_usage: u64,
    pub energy_consumption: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StrategyConfig {
    pub device_assignments: Vec<DeviceAssignment>,
    pub communication_cost: f64,
    pub estimated_total_time: f64,
    pub estimated_energy: f64,
    pub memory_peak: HashMap<String, u64>,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            device_assignments: Vec::new(),
            communication_cost: 0.0,
            estimated_total_time: 0.0,
            estimated_energy: 0.0,
            memory_peak: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ResourceUtilization {
    pub cpu: f64,
    pub memory: f64,
    pub gpu: HashMap<String, DeviceUtilization>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeviceUtilization {
    pub utilization: f64,
    pub memory: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EnergyMetrics {
    pub total_energy_joules: f64,
    pub total_energy_watt_hours: f64,
    pub device_energy: HashMap<String, DeviceEnergy>,
    pub timestamp: DateTime<Utc>,
}

impl Default for EnergyMetrics {
    fn default() -> Self {
        Self {
            total_energy_joules: 0.0,
            total_energy_watt_hours: 0.0,
            device_energy: HashMap::new(),
            timestamp: Utc::now(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DeviceEnergy {
    pub avg_power_watts: f64,
    pub max_power_watts: f64,
    pub min_power_watts: f64,
    pub energy_joules: f64,
    pub energy_watt_hours: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceSnapshot {
    pub timestamp: DateTime<Utc>,
    pub execution_time: f64,
    pub device_utilization: HashMap<String, f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CurrentMetrics {
    pub avg_execution_time: f64,
    pub devices: Vec<DeviceMetrics>,
    pub operations: Vec<OperationMetrics>,
    pub resource_utilization: ResourceUtilization,
    pub device_utilization: HashMap<String, f64>,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ComparisonDataPoint {
    pub timestamp: f64,
    pub device_type: String,  // "cpu" or "gpu"
    pub version: String,      // "baseline" or "optimized"
    pub utilization: f64,     // Percentage
    pub memory_usage: u64,    // Bytes
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ComparisonResult {
    pub data_points: Vec<ComparisonDataPoint>,
    pub baseline_avg_cpu: f64,
    pub baseline_avg_gpu: f64,
    pub optimized_avg_cpu: f64,
    pub optimized_avg_gpu: f64,
    pub cpu_improvement: f64,  // Percentage improvement
    pub gpu_improvement: f64,  // Percentage improvement
    pub completed: bool,
}

impl Default for ComparisonResult {
    fn default() -> Self {
        Self {
            data_points: Vec::new(),
            baseline_avg_cpu: 0.0,
            baseline_avg_gpu: 0.0,
            optimized_avg_cpu: 0.0,
            optimized_avg_gpu: 0.0,
            cpu_improvement: 0.0,
            gpu_improvement: 0.0,
            completed: false,
        }
    }
}
