# Instructions for Claude — SEA-NET Project

Read this file fully before doing any work in this project. These rules override any default behavior and must be followed exactly.

## 1. Teaching style
When I ask a question or give a task, explain things like a teacher. After you fix or write something, walk me through it step by step so I understand what changed and why.

## 2. Simple English
Use simple, easy-to-understand English. Avoid fancy words or long, complicated sentences.

## 3. Never run training or smoke test commands yourself
Do not run commands like model training, smoke tests, or any long-running scripts.
Instead, give me the exact command to run. I will copy and paste it into my own terminal so I can see the full output myself.

## 4. Give git commands for every update
After every code change, give me the related git commands (`git add`, `git commit`, etc.) so I can copy and paste them into my terminal myself. Do not run git commands on my behalf unless I explicitly ask you to.

## 5. Two run environments
This project runs in two different places:
- **Local machine**: VS Code on Windows. Used only for writing code and running small smoke tests.
- **Grid5000 servers** (Lille, Sophia, etc.): Used for actual GPU training runs.

Keep this in mind when writing or suggesting code: smoke tests happen locally first, and the real training scripts run later on the server.

## 6. Comments must sound student-written
Write code comments the way a student would write them, not the way an AI would. Keep them natural, short, and simple — not overly formal or robotic.

## 7. Explain libraries and complex logic like a teacher
If a piece of code uses a library or has complex logic, explain it like a teacher would: what it does, why we are using it here, and what purpose it serves in the project.

## 8. Prefer simple logic over clever logic
Always try the simple approach first. Avoid complex or fancy logic that takes a long time to understand.
The end goal is for me to fully understand everything written in this project — not just to make the code work ("vibe coding" is not the goal).
