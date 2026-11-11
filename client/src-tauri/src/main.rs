// Prevents additional console window on Windows in release
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
struct BackendStatus {
    running: bool,
    port: u16,
    url: String,
}

#[tauri::command]
fn get_backend_status() -> BackendStatus {
    BackendStatus {
        running: true,
        port: 8000,
        url: "http://127.0.0.1:8000".into(),
    }
}

#[tauri::command]
fn ping_engine() -> String {
    "Project Antimatter Sovereign Desktop Shell v0.1.0 Ready".into()
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![get_backend_status, ping_engine])
        .run(tauri::generate_context!())
        .expect("Error while running Antimatter Tauri application");
}
