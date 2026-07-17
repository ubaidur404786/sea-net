# Git Guide — SEA-NET Project

A simple reference for the git commands we use most, plus how to fix a merge conflict. Keep this updated when we learn something new.

---

## 1. What is a merge conflict?

A merge conflict happens when git cannot automatically combine two versions of the same file — for example, when you pull new changes from GitHub but you also changed the same lines locally.

Git marks the conflicting part of the file with special lines:

```
this is your local version of the code
```

- Everything between `<<<<<<< HEAD` and `=======` is **your version**.
- Everything between `=======` and `>>>>>>> branch-name` is **the other version**.
- You must delete the marker lines (`<<<<<<<`, `=======`, `>>>>>>>`) and keep only the code you actually want.

---

## 2. How to fix a merge conflict in VS Code

**Step 1 — See which files have conflicts**

```
git status
```

Files listed as `both modified` have conflicts.

**Step 2 — Open the file in VS Code**

VS Code detects the conflict markers automatically and shows colored blocks with clickable links right above each conflict:

```
Accept Current Change | Accept Incoming Change | Accept Both Changes | Compare Changes
```

- **Accept Current Change** → keeps your local version, deletes the other.
- **Accept Incoming Change** → keeps the version that came from git, deletes yours.
- **Accept Both Changes** → keeps both, one after another (you still may need to clean it up).
- **Compare Changes** → opens a side-by-side view so you can see both versions clearly before deciding.

Click the option you want for each conflict block in the file.

**Step 3 — If you prefer doing it by hand (no clicking)**

Just edit the file like a normal text file:

1. Read the two versions between the markers.
2. Decide what the final code should look like.
3. Delete the `<<<<<<<`, `=======`, and `>>>>>>>` lines completely.
4. Save the file (`Ctrl + S`).

**Step 4 — Tell git the conflict is resolved**

```
git add seanet/report.py
```

(repeat `git add <filename>` for every file you fixed)

**Step 5 — Finish the merge**

```
git commit -m "Resolve merge conflict in seanet/report.py"
```

Because you already used `git add`, git will auto-fill a merge commit message — you can just save it.

---

## 3. If a strange black terminal editor opens (Vim)

Sometimes `git commit` (without `-m`) opens an old-style editor called **Vim** instead of VS Code. It looks like a plain black screen and normal typing does nothing — don't panic, here is how to get out:

| Key                | What it does                                            |
| ------------------ | ------------------------------------------------------- |
| `i`                | Enter "insert mode" so you can type your commit message |
| `Esc`              | Leave insert mode (stop typing)                         |
| `:wq` then `Enter` | **Save and quit** (use this to finish the commit)       |
| `:q!` then `Enter` | **Quit without saving** (cancels the commit)            |

Tip: to avoid Vim entirely, always commit with a message included, like:

```
git commit -m "your message here"
```

---

## 4. Common git commands (with examples)

### Check status

```
git status
```

Shows what changed, what's staged, and what's untracked. Run this often — it's always safe.

### See the actual changes (not just filenames)

```
git diff
```

Shows line-by-line changes that are not staged yet.

```
git diff seanet/results.py
```

Shows changes for one specific file only.

### Stage changes (mark files to be committed)

```
git add main.py
```

Stages one file.

```
git add .
```

Stages everything changed in the current folder. ⚠️ Use carefully — it can accidentally add files you didn't mean to (like large data files or secrets). Prefer adding files by name.

### Commit (save a snapshot with a message)

```
git commit -m "Fix bug in pooling head config"
```

### See commit history

```
git log --oneline
```

Shows a short list of past commits, newest first. Example output:

```
40be846 add results/SEA_NET/best_results.csv
39d2fdb new branch initial seanetv1
1ef80f4 benchmark check comparison
```

### Push your commits to GitHub

```
git push
```

### Pull new commits from GitHub into your local branch

```
git pull
```

If it says "fast-forward", it means it merged cleanly with no conflicts.

### Branches

```
git branch
```

Lists all local branches, `*` marks the one you're on.

```
git checkout -b new-feature-name
```

Creates a new branch and switches to it.

```
git checkout seanetv1
```

Switches back to the `seanetv1` branch.

### Stash (temporarily save unfinished changes without committing)

```
git stash
```

Hides your uncommitted changes so your folder looks clean.

```
git stash pop
```

Brings those hidden changes back.

Useful when you need to `git pull` but have unfinished edits in the way.

### See where the remote (GitHub) is pointing

```
git remote -v
```

---

## 5. Golden safety rules

- Run `git status` before anything risky (pulling, switching branches, resetting).
- Never use `git reset --hard`, `git push --force`, or `git checkout -- .` unless you fully understand you will lose those changes forever.
- Commit small and often, with clear messages — easier to undo a small mistake than a giant one.
- If a merge conflict looks confusing, stop and ask — don't guess and force-save.

If you want to accept your local version of both files:

git checkout --ours seanet/report.py
git checkout --ours seanet/results.py
git add seanet/report.py seanet/results.py

Or, if you want to accept the incoming/stashed version:

git checkout --theirs seanet/report.py
git checkout --theirs seanet/results.py
git add seanet/report.py seanet/results.py
