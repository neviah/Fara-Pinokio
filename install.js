#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { spawn, exec } = require('child_process');

/**
 * Pinokio Install Script for Fara-7B Gradio Interface
 * This script sets up the complete environment for running Fara-7B with Gradio
 */

class FaraInstaller {
    constructor() {
        this.isWindows = process.platform === 'win32';
        this.pythonCmd = this.isWindows ? 'python' : 'python3';
        this.pipCmd = 'pip';
        this.projectDir = process.cwd();
        this.venvDir = path.join(this.projectDir, 'venv');
    }

    log(message) {
        console.log(`[FARA-INSTALL] ${message}`);
    }

    error(message) {
        console.error(`[FARA-ERROR] ${message}`);
    }

    async runCommand(command, args = [], options = {}) {
        return new Promise((resolve, reject) => {
            const child = spawn(command, args, { 
                stdio: 'inherit', 
                shell: true,
                ...options 
            });
            
            child.on('close', (code) => {
                if (code === 0) {
                    resolve();
                } else {
                    reject(new Error(`Command failed with code ${code}`));
                }
            });
            
            child.on('error', (error) => {
                reject(error);
            });
        });
    }

    async checkPython() {
        this.log('Checking Python installation...');
        try {
            await this.runCommand(this.pythonCmd, ['--version']);
            this.log('✅ Python found');
            return true;
        } catch (error) {
            this.error('❌ Python not found. Please install Python 3.8+ first.');
            return false;
        }
    }

    async createVirtualEnvironment() {
        this.log('Creating virtual environment...');
        try {
            if (fs.existsSync(this.venvDir)) {
                this.log('Virtual environment already exists, removing...');
                fs.rmSync(this.venvDir, { recursive: true, force: true });
            }
            
            await this.runCommand(this.pythonCmd, ['-m', 'venv', this.venvDir]);
            this.log('✅ Virtual environment created');
            
            // Update pip and pip commands for virtual environment
            if (this.isWindows) {
                this.pipCmd = path.join(this.venvDir, 'Scripts', 'pip.exe');
                this.pythonCmd = path.join(this.venvDir, 'Scripts', 'python.exe');
            } else {
                this.pipCmd = path.join(this.venvDir, 'bin', 'pip');
                this.pythonCmd = path.join(this.venvDir, 'bin', 'python');
            }
            
            return true;
        } catch (error) {
            this.error(`❌ Failed to create virtual environment: ${error.message}`);
            return false;
        }
    }

    async upgradePip() {
        this.log('Upgrading pip...');
        try {
            await this.runCommand(this.pipCmd, ['install', '--upgrade', 'pip']);
            this.log('✅ Pip upgraded');
            return true;
        } catch (error) {
            this.error(`❌ Failed to upgrade pip: ${error.message}`);
            return false;
        }
    }

    async cloneFaraRepository() {
        this.log('Cloning Fara repository...');
        const faraDir = path.join(this.projectDir, 'fara');
        
        try {
            if (fs.existsSync(faraDir)) {
                this.log('Fara repository already exists, updating...');
                await this.runCommand('git', ['pull'], { cwd: faraDir });
            } else {
                await this.runCommand('git', [
                    'clone', 
                    'https://github.com/microsoft/fara.git',
                    faraDir
                ]);
            }
            this.log('✅ Fara repository ready');
            return true;
        } catch (error) {
            this.error(`❌ Failed to clone repository: ${error.message}`);
            return false;
        }
    }

    async installFara() {
        this.log('Installing Fara package...');
        const faraDir = path.join(this.projectDir, 'fara');
        
        try {
            // Install Fara in editable mode
            await this.runCommand(this.pipCmd, ['install', '-e', faraDir]);
            this.log('✅ Fara package installed');
            return true;
        } catch (error) {
            this.error(`❌ Failed to install Fara: ${error.message}`);
            return false;
        }
    }

    async installDependencies() {
        this.log('Installing Python dependencies...');
        
        const dependencies = [
            'gradio>=4.0.0',
            'playwright>=1.40.0',
            'openai>=1.0.0',
            'Pillow>=9.0.0',
            'tenacity>=8.0.0',
            'numpy>=1.20.0',
            'torch>=2.0.0',  // For potential local model inference
            'transformers>=4.30.0',  // For tokenization if needed
        ];

        try {
            for (const dep of dependencies) {
                this.log(`Installing ${dep}...`);
                await this.runCommand(this.pipCmd, ['install', dep]);
            }
            this.log('✅ Python dependencies installed');
            return true;
        } catch (error) {
            this.error(`❌ Failed to install dependencies: ${error.message}`);
            return false;
        }
    }

    async installPlaywrightBrowsers() {
        this.log('Installing Playwright browsers...');
        try {
            await this.runCommand(this.pythonCmd, ['-m', 'playwright', 'install']);
            this.log('✅ Playwright browsers installed');
            return true;
        } catch (error) {
            this.error(`❌ Failed to install Playwright browsers: ${error.message}`);
            return false;
        }
    }

    async createConfigFiles() {
        this.log('Creating configuration files...');
        
        try {
            // Create default endpoint config
            const endpointConfig = {
                model: "microsoft/Fara-7B",
                base_url: "http://localhost:5000/v1",
                api_key: "not-needed"
            };
            
            const configPath = path.join(this.projectDir, 'endpoint_config.json');
            fs.writeFileSync(configPath, JSON.stringify(endpointConfig, null, 2));
            
            // Create .env file template
            const envContent = `# Fara-7B Configuration
# Uncomment and set if using BrowserBase
# BROWSERBASE_API_KEY=your_api_key_here
# BROWSERBASE_PROJECT_ID=your_project_id_here

# Uncomment if using Azure Foundry
# AZURE_OPENAI_ENDPOINT=your_endpoint_here
# AZURE_OPENAI_API_KEY=your_key_here
`;
            
            const envPath = path.join(this.projectDir, '.env.example');
            fs.writeFileSync(envPath, envContent);
            
            this.log('✅ Configuration files created');
            return true;
        } catch (error) {
            this.error(`❌ Failed to create config files: ${error.message}`);
            return false;
        }
    }

    async createLaunchScript() {
        this.log('Creating launch script...');
        
        try {
            const scriptContent = this.isWindows ? 
                `@echo off
echo Starting Fara-7B Gradio Interface...
cd /d "%~dp0"
call venv\\Scripts\\activate
python gradio_interface.py
pause` :
                `#!/bin/bash
echo "Starting Fara-7B Gradio Interface..."
cd "$(dirname "$0")"
source venv/bin/activate
python3 gradio_interface.py`;
            
            const scriptPath = this.isWindows ? 
                path.join(this.projectDir, 'start_fara.bat') :
                path.join(this.projectDir, 'start_fara.sh');
            
            fs.writeFileSync(scriptPath, scriptContent);
            
            if (!this.isWindows) {
                // Make shell script executable
                await this.runCommand('chmod', ['+x', scriptPath]);
            }
            
            this.log('✅ Launch script created');
            return true;
        } catch (error) {
            this.error(`❌ Failed to create launch script: ${error.message}`);
            return false;
        }
    }

    async performInstallation() {
        this.log('🚀 Starting Fara-7B installation for Pinokio...');
        
        const steps = [
            () => this.checkPython(),
            () => this.createVirtualEnvironment(),
            () => this.upgradePip(),
            () => this.cloneFaraRepository(),
            () => this.installFara(),
            () => this.installDependencies(),
            () => this.installPlaywrightBrowsers(),
            () => this.createConfigFiles(),
            () => this.createLaunchScript()
        ];

        for (let i = 0; i < steps.length; i++) {
            this.log(`Step ${i + 1}/${steps.length}...`);
            const success = await steps[i]();
            if (!success) {
                this.error('Installation failed!');
                process.exit(1);
            }
        }

        this.log('🎉 Installation completed successfully!');
        this.log('');
        this.log('📋 Next steps:');
        this.log('1. Start a VLLM server: vllm serve "microsoft/Fara-7B" --port 5000 --dtype auto');
        this.log('2. Run the Gradio interface with: node pinokio.js');
        this.log('3. Open http://localhost:7860 in your browser');
        this.log('');
        this.log('💡 For Azure Foundry deployment, update endpoint_config.json with your credentials');
    }
}

// Main execution
if (require.main === module) {
    const installer = new FaraInstaller();
    installer.performInstallation().catch((error) => {
        console.error('❌ Installation failed:', error.message);
        process.exit(1);
    });
}

module.exports = FaraInstaller;