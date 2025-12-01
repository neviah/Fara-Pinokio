module.exports = {
  run: [
    // Windows NVIDIA CUDA
    {
      when: "{{platform === 'win32' && gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        message: [
          "python -m pip install --upgrade pip",
          // Official pytorch pip with CUDA 12.1 support
          "pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio"
        ],
        venv: "env"
      }
    },
    // Windows AMD/Intel (CPU fallback)
    {
      when: "{{platform === 'win32' && gpu !== 'nvidia'}}",
      method: "shell.run",
      params: {
        message: [
          "python -m pip install --upgrade pip",
          "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
        ],
        venv: "env"
      }
    },
    // Linux NVIDIA CUDA
    {
      when: "{{platform === 'linux' && gpu === 'nvidia'}}",
      method: "shell.run",
      params: {
        message: [
          "python -m pip install --upgrade pip",
          "pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio"
        ],
        venv: "env"
      }
    },
    // macOS (CPU only)
    {
      when: "{{platform === 'darwin'}}",
      method: "shell.run",
      params: {
        message: [
          "python -m pip install --upgrade pip",
          "pip install torch torchvision torchaudio"
        ],
        venv: "env"
      }
    },
    {
      method: "input",
      params: {
        title: "Torch Installation",
        description: "Torch has been installed based on your platform. Return to the dashboard and Start the app."
      }
    }
  ]
}
