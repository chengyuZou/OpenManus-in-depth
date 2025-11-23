[OpenManus源地址](https://github.com/FoundationAgents/OpenManus)

<details>
<summary>📅 更新日志 (Update Log)</summary>

**2025.11.20 开源**
- llm.py 
</details>

完成情况(打勾为完成，没打勾为没完成)
```text
OpenManus/
├── main.py
├── run_flow.py
├── run_mcp.py
├── run_mcp_server.py
├── sandbox_main.py
├── setup.py
├── requirements.txt
├── app/
│   ├── __init__.py
│   ├── bedrock.py
│   ├── config.py
│   ├── exceptions.py
│   ├── llm.py  ✅
│   ├── logger.py
│   ├── schema.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── browser.py
│   │   ├── data_analysis.py
│   │   ├── manus.py
│   │   ├── mcp.py
│   │   ├── react.py
│   │   ├── sandbox_agent.py
│   │   └── toolcall.py
│   ├── tool/
│   │   ├── __init__.py
│   │   ├── ask_human.py
│   │   ├── base.py
│   │   ├── bash.py
│   │   ├── browser_use_tool.py
│   │   ├── computer_use_tool.py
│   │   ├── crawl4ai.py
│   │   ├── create_chat_completion.py
│   │   ├── file_operators.py
│   │   ├── mcp.py
│   │   ├── planning.py
│   │   ├── python_execute.py
│   │   ├── str_replace_editor.py
│   │   ├── terminate.py
│   │   ├── tool_collection.py
│   │   ├── web_search.py
│   │   ├── search/
│   │   │   ├── __init__.py
│   │   │   ├── baidu_search.py
│   │   │   ├── base.py
│   │   │   ├── bing_search.py
│   │   │   ├── duckduckgo_search.py
│   │   │   └── google_search.py
│   │   ├── chart_visualization/
│   │   │   ├── README.md
│   │   │   ├── README_ja.md
│   │   │   ├── README_ko.md
│   │   │   ├── README_zh.md
│   │   │   ├── __init__.py
│   │   │   ├── chart_prepare.py
│   │   │   ├── data_visualization.py
│   │   │   ├── package-lock.json
│   │   │   ├── package.json
│   │   │   ├── python_execute.py
│   │   │   ├── src/
│   │   │   └── test/
│   │   └── sandbox/
│   │       ├── sb_browser_tool.py
│   │       ├── sb_files_tool.py
│   │       ├── sb_shell_tool.py
│   │       └── sb_vision_tool.py
│   ├── prompt/
│   │   ├── __init__.py
│   │   ├── browser.py
│   │   ├── manus.py
│   │   ├── mcp.py
│   │   ├── planning.py
│   │   ├── swe.py
│   │   ├── toolcall.py
│   │   └── visualization.py
│   ├── mcp/
│   │   ├── __init__.py
│   │   └── server.py
│   ├── flow/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── flow_factory.py
│   │   └── planning.py
│   ├── sandbox/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   └── core/
│   │       ├── exceptions.py
│   │       ├── manager.py
│   │       ├── sandbox.py
│   │       └── terminal.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── files_utils.py
│   │   └── logger.py
│   └── daytona/
│       ├── README.md
│       ├── sandbox.py
│       └── tool_base.py
├── config/
├── workspace/
├── examples/
├── tests/
└── protocol/
```
