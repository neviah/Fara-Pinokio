module.exports = {
  "title": "Fara-7B Computer Use Agent",
  "description": "Microsoft's efficient 7B parameter agentic model for automating web tasks",
  "icon": "icon.png",
  "menu": async (kernel) => {
    return [
      {
        "text": "Install",
        "href": "install"
      },
      {
        "text": "Start", 
        "href": "start"
      }
    ]
  }
}
}