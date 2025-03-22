use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde_json::json;
use std::time::Duration;

use crate::data::{
    CurrentMetrics, EnergyMetrics, StrategyConfig, ComparisonResult
};

pub struct PythonFrameworkClient {
    client: Client,
    base_url: String,
}

impl PythonFrameworkClient {
    pub fn new(base_url: &str) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(10))
            .build()?;
        
        Ok(Self {
            client,
            base_url: base_url.to_string(),
        })
    }
    
    pub fn get_current_metrics(&self) -> Result<CurrentMetrics> {
        // In a real implementation, this would make an HTTP request to the Python API
        // For the purpose of this example, let's return mock data
        #[cfg(feature = "mock_api")]
        {
            use rand::Rng;
            use chrono::Utc;
            use std::collections::HashMap;
            
            let mut rng = rand::thread_rng();
            
            // Mock GPU device
            let gpu = DeviceMetrics {
                device_id: "cuda:0".to_string(),
                name: "NVIDIA GeForce RTX 3080".to_string(),
                utilization: rng.gen_range(30.0..90.0),
                memory_used: rng.gen_range(2_000_000_000..8_000_000_000),
                memory_total: 10_000_000_000,
                operations_count: rng.gen_range(50..200),
                compute_time: rng.gen_range(0.01..0.05),
            };
            
            // Mock CPU device
            let cpu = DeviceMetrics {
                device_id: "cpu:0".to_string(),
                name: "Intel Core i9-10900K".to_string(),
                utilization: rng.gen_range(10.0..50.0),
                memory_used: rng.gen_range(4_000_000_000..12_000_000_000),
                memory_total: 16_000_000_000,
                operations_count: rng.gen_range(20..100),
                compute_time: rng.gen_range(0.05..0.2),
            };
            
            // Mock operations
            let operations = vec![
                OperationMetrics {
                    op_id: "conv1".to_string(),
                    device_id: "cuda:0".to_string(),
                    execution_time: rng.gen_range(0.001..0.01),
                    memory_usage: rng.gen_range(100_000_000..500_000_000),
                    flops: rng.gen_range(10_000_000..100_000_000),
                },
                OperationMetrics {
                    op_id: "relu1".to_string(),
                    device_id: "cuda:0".to_string(),
                    execution_time: rng.gen_range(0.0001..0.001),
                    memory_usage: rng.gen_range(10_000_000..50_000_000),
                    flops: rng.gen_range(1_000_000..10_000_000),
                },
                OperationMetrics {
                    op_id: "fc1".to_string(),
                    device_id: "cpu:0".to_string(),
                    execution_time: rng.gen_range(0.005..0.02),
                    memory_usage: rng.gen_range(200_000_000..800_000_000),
                    flops: rng.gen_range(5_000_000..50_000_000),
                },
            ];
            
            // Mock resource utilization
            let mut gpu_utils = HashMap::new();
            gpu_utils.insert("cuda:0".to_string(), DeviceUtilization {
                utilization: rng.gen_range(30.0..90.0),
                memory: rng.gen_range(30.0..90.0),
            });
            
            let resource_utilization = ResourceUtilization {
                cpu: rng.gen_range(10.0..50.0),
                memory: rng.gen_range(40.0..80.0),
                gpu: gpu_utils,
            };
            
            // Mock device utilization
            let mut device_utilization = HashMap::new();
            device_utilization.insert("cuda:0".to_string(), rng.gen_range(30.0..90.0));
            device_utilization.insert("cpu:0".to_string(), rng.gen_range(10.0..50.0));
            
            return Ok(CurrentMetrics {
                avg_execution_time: rng.gen_range(0.01..0.05),
                devices: vec![gpu, cpu],
                operations,
                resource_utilization,
                device_utilization,
                timestamp: Utc::now(),
            });
        }
        
        // Actual implementation for communicating with Python API
        let url = format!("{}/metrics/current", self.base_url);
        let response = self.client.get(&url).send()?;
        let metrics = response.json::<CurrentMetrics>()?;
        Ok(metrics)
    }
    
    pub fn get_energy_metrics(&self) -> Result<EnergyMetrics> {
        let url = format!("{}/metrics/energy", self.base_url);
        let response = self.client.get(&url).send()?;
        let metrics = response.json::<EnergyMetrics>()?;
        Ok(metrics)
    }
    
    pub fn get_current_strategy(&self) -> Result<StrategyConfig> {
        let url = format!("{}/strategy/current", self.base_url);
        let response = self.client.get(&url).send()?;
        let strategy = response.json::<StrategyConfig>()?;
        Ok(strategy)
    }
    
    pub fn update_config(&self, key: &str, value: &str) -> Result<()> {
        let url = format!("{}/config/update", self.base_url);
        let payload = json!({
            "key": key,
            "value": value,
        });
        
        let _response = self.client.post(&url)
            .json(&payload)
            .send()?;
        
        Ok(())
    }
    
    pub fn run_comparison(&self) -> Result<ComparisonResult> {
        let url = format!("{}/comparison/run", self.base_url);
        let response = self.client.post(&url).send()?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to run comparison: {}", response.status()));
        }
        
        let comparison_data = response.json::<ComparisonResult>()?;
        Ok(comparison_data)
    }
    
    pub fn get_comparison_status(&self) -> Result<ComparisonResult> {
        let url = format!("{}/comparison/status", self.base_url);
        let response = self.client.get(&url).send()?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to get comparison status: {}", response.status()));
        }
        
        let status = response.json::<ComparisonResult>()?;
        Ok(status)
    }
    
    pub fn save_comparison_results(&self, filename: &str) -> Result<()> {
        let url = format!("{}/comparison/save", self.base_url);
        let payload = json!({
            "filename": filename,
        });
        
        let response = self.client.post(&url)
            .json(&payload)
            .send()?;
        
        if !response.status().is_success() {
            return Err(anyhow::anyhow!("Failed to save comparison results: {}", response.status()));
        }
        
        Ok(())
    }
}
