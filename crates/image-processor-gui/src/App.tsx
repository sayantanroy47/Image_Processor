import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/tauri";
import "./App.css";

interface SystemInfo {
  version: string;
  platform: string;
  architecture: string;
  cpu_cores: number;
}

interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
}

function App() {
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);
  const [capabilities, setCapabilities] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadSystemInfo();
    loadCapabilities();
  }, []);

  async function loadSystemInfo() {
    try {
      const response: ApiResponse<SystemInfo> = await invoke("get_system_info");
      if (response.success && response.data) {
        setSystemInfo(response.data);
      }
    } catch (error) {
      console.error("Failed to load system info:", error);
    }
  }

  async function loadCapabilities() {
    try {
      const response: ApiResponse<string[]> = await invoke("get_capabilities");
      if (response.success && response.data) {
        setCapabilities(response.data);
      }
    } catch (error) {
      console.error("Failed to load capabilities:", error);
    } finally {
      setLoading(false);
    }
  }

  async function handleConvert() {
    try {
      const response: ApiResponse<string> = await invoke("convert_image", {
        inputPath: "/path/to/input.jpg",
        outputPath: "/path/to/output.png",
        format: "png",
        quality: 85,
      });
      
      if (response.success) {
        alert("Conversion initiated: " + response.data);
      } else {
        alert("Conversion failed: " + response.error);
      }
    } catch (error) {
      console.error("Conversion error:", error);
      alert("Conversion error: " + error);
    }
  }

  if (loading) {
    return (
      <div className="container">
        <h1>Loading Image Processor...</h1>
      </div>
    );
  }

  return (
    <div className="container">
      <h1>Image Processor</h1>
      
      {systemInfo && (
        <div className="system-info">
          <h2>System Information</h2>
          <p><strong>Version:</strong> {systemInfo.version}</p>
          <p><strong>Platform:</strong> {systemInfo.platform}</p>
          <p><strong>Architecture:</strong> {systemInfo.architecture}</p>
          <p><strong>CPU Cores:</strong> {systemInfo.cpu_cores}</p>
        </div>
      )}

      <div className="capabilities">
        <h2>Available Features</h2>
        <ul>
          {capabilities.map((capability, index) => (
            <li key={index}>{capability}</li>
          ))}
        </ul>
      </div>

      <div className="actions">
        <h2>Quick Actions</h2>
        <button onClick={handleConvert}>
          Test Image Conversion
        </button>
        <p className="note">
          Note: Full functionality will be implemented in future tasks.
          This is a foundation setup.
        </p>
      </div>
    </div>
  );
}

export default App;