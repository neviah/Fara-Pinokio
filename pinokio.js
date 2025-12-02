module.exports = {
  version: "3.7",
  title: "Fara-7B Computer Use Agent", 
  description: "Microsoft's 7B parameter computer use agent with Gradio interface",
  icon: "icon.svg",
  menu: async (kernel, info) => {
    let installed = info.exists("env")
    let running = {
      install: info.running("install.json"),
      start: info.running("start.json")
    }
    if (running.install) {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Installing",
        href: "install.json",
      }]
    } else if (installed) {
      if (running.start) {
        return [{
          default: true,
          icon: "fa-solid fa-rocket",
          text: "Open Web UI",
          href: "http://127.0.0.1:7860",
        }, {
          icon: 'fa-solid fa-terminal',
          text: "Terminal",
          href: "start.json",
        }]
      } else {
        return [{
          default: true,
          icon: "fa-solid fa-power-off",
          text: "Start",
          href: "start.json",
        }, {
          icon: "fa-solid fa-plug",
          text: "Install",
          href: "install.json",
        }]
      }
    } else {
      return [{
        default: true,
        icon: "fa-solid fa-plug",
        text: "Install",
        href: "install.json",
      }]
    }
  }
}