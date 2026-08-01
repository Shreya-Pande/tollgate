# Tollgate — Phase Guide

CampusBook jaisa hi format. Har phase ke end mein ek check hai — wo pass ho jaaye tabhi aage badhna.

**Storage plan:** project + saara cache **E drive** pe. C drive pe sirf teen cheezein: Python, Git, Claude Code (~650 MB total).

---

## Storage ka hisaab

| Kya | Kahan | Size |
|---|---|---|
| Python 3.12 | `E:\Python312` | ~150 MB |
| Git for Windows | C (default) | ~350 MB |
| Claude Code | C (user folder) | ~150 MB |
| venv + packages | `E:\Tollgate\.venv` | ~700 MB |
| Embedding model | `E:\hf-cache` | ~90 MB |
| Datasets | `E:\Tollgate\data` | ~150 MB |

**Jo install nahi kar rahe:** Docker Desktop (~4 GB), WSL 2 (~2 GB), PyTorch (~2.5 GB), local Postgres. Sab ka alternative neeche hai.

---

# PHASE 0 — Installs
### 0% → 8%

### 0.1 Python 3.12

python.org/downloads se lo. Installer chalao, phir:

- **"Add python.exe to PATH"** pe tick zaroor karo
- **"Customize installation"** pe click karo → Next → install location badal ke **`E:\Python312`** kar do

3.12 hi lena, 3.13 nahi — kuch ML packages abhi 3.13 pe theek se nahi chalte.

### 0.2 Git for Windows

git-scm.com/downloads/win — sab default rakho, bas ek jagah **"Checkout as-is, commit Unix-style line endings"** chuno.

### 0.3 Claude Code

PowerShell kholo (admin ki zaroorat nahi):

```powershell
irm https://claude.ai/install.ps1 | iex
```

Node.js ki zaroorat nahi hai, ye khud ka binary laata hai.

⚠️ **Note:** Claude Code ke liye paid plan chahiye (Pro / Max / Team) ya Console API key. Free plan mein nahi milta. Agar abhi nahi hai toh Phase 5 tak decide kar lena — waise tak kaam nahi rukega.

### 0.4 VS Code extensions

Python, Pylance, Ruff. Bas. Docker extension ki zaroorat nahi kyunki Docker install hi nahi kar rahe.

### ✅ Check

Nayi PowerShell window kholo (purani mein PATH update nahi hoga):

```powershell
python --version     # 3.12.x
git --version        # 2.x
claude --version     # 2.1.x
```

Agar `python` likhne pe Microsoft Store khul jaaye — PATH wala tick reh gaya tha. Installer dubara chalao → **Modify** → Add to PATH.

---

# PHASE 1 — Accounts
### 8% → 12%

Sirf do banane hain abhi:

**GitHub** — repo aur CI ke liye. Free.

**Neon** (neon.tech) — free Postgres, aur isme pgvector already available hai. Project banao, uske SQL editor mein ye chalao:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
SELECT '[1,2,3]'::vector <=> '[1,2,4]'::vector;
```

Connection string copy karke kahin safe rakh lo.

**LLM provider key** — jo bhi use kar rahe ho. **Dashboard mein spend limit abhi set kar do.** Batch jobs chalenge hazaaron prompts pe; ek loop bug raat ko paisa kha sakta hai.

Fly.io / Render abhi mat banao — wo Phase 10 mein chahiye.

### ✅ Check
Neon ka `<=>` query ek chhota number return kare. Isse pata chal gaya ki database, extension aur distance operator — teeno kaam kar rahe hain.

---

# PHASE 2 — Folder aur Git
### 12% → 20%

```powershell
cd E:\
mkdir Tollgate; cd Tollgate
git init
```

Folders:

```powershell
mkdir app, eval, migrations, tests, requirements, scripts, data
mkdir eval\figures, data\raw
```

Khaali files:

```powershell
ni app\__init__.py, app\main.py, app\config.py, app\db.py, app\normalize.py, `
   app\constraints.py, app\embedder.py, app\cache.py, app\upstream.py -ItemType File
ni eval\build_dataset.py, eval\run_sweep.py, eval\cost_model.py, eval\bench_gptcache.py -ItemType File
ni migrations\001_init.sql, migrations\run.py -ItemType File
ni tests\test_constraints.py, tests\test_admission.py, `
   tests\test_tenant_isolation.py, tests\test_cache_key.py -ItemType File
ni scripts\fetch_data.py, numbers.md, .env, .gitignore, Dockerfile -ItemType File
```

Git config (ek baar, global):

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
git config --global core.autocrlf input
```

`core.autocrlf input` important hai — warna Windows ke line endings files mein ghus jaate hain aur baad mein Linux pe weird errors aate hain.

`.gitignore` mein ye daalo — **pehle commit se pehle**:

```
.venv/
.venv-gptcache/
__pycache__/
*.pyc
.env
data/
eval/figures/*.png
.pytest_cache/
.ruff_cache/
```

`.env` mein:

```
DATABASE_URL=<Neon connection string>
UPSTREAM_API_KEY=
UPSTREAM_BASE_URL=
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
SIMILARITY_THRESHOLD=0.95
TOP_K=3
WORKLOAD_LABEL=mixed
```

### ✅ Check
`.gitignore` mein `.env` likha hua hai — commit karne se pehle confirm karo. API key git history mein chali gayi toh nikalna bada painful hai, aur interviewer repo clone karega.

---

# PHASE 3 — Cache redirect + venv
### 20% → 30%

**Ye phase skip mat karna.** Yehi wo jagah hai jahan chupke se C drive bhar jaata hai.

Pehle cache locations E pe bhejo (permanent, ek baar):

```powershell
[Environment]::SetEnvironmentVariable("PIP_CACHE_DIR", "E:\pip-cache", "User")
[Environment]::SetEnvironmentVariable("HF_HOME", "E:\hf-cache", "User")
[Environment]::SetEnvironmentVariable("FASTEMBED_CACHE_PATH", "E:\fastembed-cache", "User")
```

**Ab PowerShell band karke nayi kholo** (env variables tabhi lagenge), aur wapas `E:\Tollgate` pe aao.

Verify:

```powershell
echo $env:HF_HOME     # E:\hf-cache aana chahiye
```

venv banao:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

⚠️ Agar *"running scripts is disabled"* aaye (ye 90% logon ko aata hai):

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

`Y` dabao, phir activate dubara chalao. Prompt mein `(.venv)` dikhna chahiye.

### requirements folder — 4 files

**`requirements\base.txt`** — jo Docker image mein jaayega:

```
fastapi
uvicorn[standard]
asyncpg
pgvector
pydantic
pydantic-settings
httpx
fastembed
python-dotenv
```

**`requirements\eval.txt`** — offline analysis, image mein nahi:

```
-r base.txt
scikit-learn
numpy
pandas
pyarrow
matplotlib
datasets
```

**`requirements\dev.txt`** — jo aap local pe install karoge:

```
-r eval.txt
pytest
pytest-asyncio
ruff
```

**`requirements\locked.txt`** — ye aap nahi likhoge, generate hoga.

> **Interview mein poocha jaa sakta hai:** teen files kyun? Kyunki serving container mein matplotlib ya datasets ki koi zaroorat nahi — image chhota rehta hai. Aur `locked.txt` alag isliye ki declared dependencies aur resolved versions do alag cheezein hain. Node mein `package.json` vs `package-lock.json` — same idea.

Install:

```powershell
python -m pip install --upgrade pip
pip install -r requirements\dev.txt
pip freeze > requirements\locked.txt
```

**~700 MB, 5-8 minute.** (PyTorch wale route mein ye 3 GB aur 20 minute hota.)

### ✅ Check

```powershell
python -c "import fastapi, asyncpg, pgvector, sklearn, datasets, fastembed; print('ok')"
```

Phir E drive pe folders check karo — `E:\pip-cache` bana hona chahiye. Agar `C:\Users\<you>\AppData\Local\pip` bana hai toh env variable set nahi hua, wapas upar jao.

Pehla commit:

```powershell
git add .
git commit -m "Scaffold: structure, requirements, gitignore"
```

GitHub pe khaali repo banao (README/gitignore mat add karna), phir:

```powershell
git remote add origin https://github.com/<you>/tollgate.git
git branch -M main
git push -u origin main
```

---

# PHASE 4 — Model + data
### 30% → 38%

`scripts\fetch_data.py`:

```python
from datasets import load_dataset
from pathlib import Path
Path("data/raw").mkdir(parents=True, exist_ok=True)

load_dataset("databricks/databricks-dolly-15k", split="train") \
    .to_parquet("data/raw/dolly.parquet")
load_dataset("paws", "labeled_final", split="train") \
    .select(range(20000)).to_parquet("data/raw/paws.parquet")
load_dataset("glue", "qqp", split="train") \
    .select(range(20000)).to_parquet("data/raw/qqp.parquet")
```

```powershell
python scripts\fetch_data.py
```

Ab embedding model aur pehla measurement:

```python
# scripts\smoke_test.py
from fastembed import TextEmbedding
import numpy as np, time

m = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
texts = ["summarize this passage in three bullet points"] * 100

list(m.embed(texts[:5]))                      # warm-up

t = time.perf_counter()
v = list(m.embed(texts))
print("per-encode ms:", (time.perf_counter() - t) / 100 * 1000)
print("dim:", len(v[0]), "norm:", np.linalg.norm(v[0]))
```

### ✅ Check
`dim: 384`, `norm` lagbhag 1.0, aur per-encode ~5-20 ms.

**Wo number `numbers.md` mein likh do.** Ye aapki latency table ki pehli line hai.

Dolly ka licence page ek baar dekh lo aur README mein note kar dena.

---

# PHASE 5 — Pehla Claude Code prompt
### 38% → 52%

Ab coding shuru. `E:\Tollgate` mein PowerShell kholo aur:

```powershell
claude
```

**Pehle `TOLLGATE.md` ko project folder mein daal do** (Section 3 ka schema aur Section 4 ka request path Claude Code ko chahiye honge).

### Pehla prompt — copy karke paste karo

```
Read TOLLGATE.md in this repo — that's the full spec. We're building Day 1
(Part II, "Day 1 — Plumbing"). Do NOT build the cache or the constraint
extractor yet. Miss path only.

Environment: Windows, Python 3.12, venv already active, Neon Postgres
(cloud, no Docker). Embeddings via fastembed, NOT sentence-transformers:

    from fastembed import TextEmbedding
    model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    vec = list(model.embed([text]))[0]      # numpy array, 384-dim, normalized

Build these five things, in this order, and stop after each so I can review:

1. migrations/001_init.sql — the schema exactly as written in TOLLGATE.md §3.
   Plus migrations/run.py: an asyncpg runner that tracks applied files in a
   _migrations table and wraps each file in a transaction.

2. app/db.py — asyncpg pool. IMPORTANT: pass init=register_vector to
   create_pool, on the pool not a single connection.

3. app/config.py — pydantic-settings reading from .env. EMBEDDING_MODEL and
   WORKLOAD_LABEL must be config values, not constants.
   app/normalize.py — raw_hash() over the RAW prompt (sha256, exact-match key)
   and normalize() for embedding input only. Read §3 for why these differ.

4. app/upstream.py — httpx async client + cost accounting. The budget charge
   must be the single atomic UPDATE ... WHERE ... RETURNING statement shown
   in §4. No read-modify-write.

5. app/main.py — FastAPI, POST /v1/chat/completions with OpenAI-compatible
   Pydantic models, GET /healthz. Flow: resolve tenant → charge budget →
   exact-match lookup on raw hash → on miss, call upstream → store
   cache_entries row → write cache_decisions row → return.
   The cache entry and the decision row MUST commit in one transaction.

No semantic search yet. Explain each design choice briefly as you go — I need
to defend all of this in interviews.
```

### Kyun ye prompt aisa hai

- **Spec file reference kiya** — 400 lines paste karne se behtar
- **fastembed explicitly bataya**, warna wo sentence-transformers likhega (spec mein wahi likha hai)
- **"stop after each"** — aap har file padh sako, ek 800-line dump na aaye
- **Do jagah "IMPORTANT"** — yehi do cheezein sabse zyada galat hoti hain
- **"explain each design choice"** — kyunki interview mein aapko bolna hai, Claude ko nahi

### ✅ Check

```powershell
python migrations\run.py      # dubara chalao — kuch print nahi hona chahiye
uvicorn app.main:app --reload
```

Dusri window mein:

```powershell
curl -X POST http://localhost:8000/v1/chat/completions `
  -H "Authorization: Bearer test-key" -H "Content-Type: application/json" `
  -d '{\"model\":\"...\",\"messages\":[{\"role\":\"user\",\"content\":\"what is a b-tree\"}]}'
```

Neon ke SQL editor mein: `SELECT decision, total_ms FROM cache_decisions;` — ek row, `MISS_NO_CANDIDATE`.

Phir **wahi request dubara bhejo** → `HIT_EXACT`, single-digit ms.

Commit.

---

# PHASE 6 — Eval script (fake data pe)
### 52% → 58%

Chhota phase, lekin **skip mat karna**. Rule ye hai: agar resume bullet ka number nikalne wali script Day 1 ke end tak nahi bani, project risk mein hai.

`eval/run_sweep.py` **khud likho** — ye interview ka core hai. `np.where(compatible, sim, -1.0)` wali trick TOLLGATE.md step 1.8 mein hai; wo samajh lo, kyunki gated rule ek plain threshold nahi hai aur wahi transform dono classifiers ko ek axis pe laata hai.

Fake random data pe chalao, ek meaningless ROC PNG nikalo. Numbers bekaar hain — pipeline asli hai.

### ✅ Check
`eval/figures/roc.png` exist karti ho, do curves dikhein.

---

# PHASE 7 — Asli data, pehla asli number
### 58% → 72%

**Ye sabse important phase hai. Isko poora din do.**

`eval/build_dataset.py` — teen populations (P1, P2, P3) + P3 controls. TOLLGATE.md steps 2.1-2.3.

**Perturbation families aur held-out split khud likho.** Aath families banao, rules sirf paanch ke liye. Agar ye Claude Code se likhwaya toh aapko yaad nahi rahega ki kaun si held-out hai aur kyun — aur interview mein yahi poocha jaayega.

Phir 100 generated pairs khud padho aur labels verify karo (~40 min). Error rate note karo — yehi "aapko kaise pata labels sahi hain" ka jawab hai.

Similarity search + cosine-only admission — ye Claude Code se karwa sakte ho.

### ✅ Check — do numbers
1. Pehla asli ROC, teeno populations alag-alag, AUC with bootstrap CI
2. Exact vs semantic hit split (step 2.7) — `SELECT decision, count(*) FROM cache_decisions GROUP BY decision;`

⚠️ **Decision point:** agar cosine-only P2 pe already clean separate kar raha hai (AUC 0.9+), toh aapka near-miss set aasan hai. **Aaj hi** PAWS filter tight karo. Ye Phase 9 mein pata chala toh time nahi bachega.

---

# PHASE 8 — Constraint gate (asli contribution)
### 72% → 84%

`app/constraints.py` — paanch dimensions. **Khud likho.** Ye project ka dil hai.

Pehle `tests/test_constraints.py` likho (failing), phir extractor. Chaho toh Claude Code ko test dekar bolo "make these pass" — aur diff dhyaan se padho.

Phir: gate ko admission loop mein wire karo, sweep dubara chalao, **curve-shift figure** banao. Plus output-constraint audit (§7 — free hai, upstream call nahi lagta) aur cost model.

### ✅ Check
Do figures: gated vs cosine-only ROC (curve upar shift honi chahiye), aur false hits kis dimension se aaye uska breakdown.

---

# PHASE 9 — Defence
### 84% → 94%

- Held-out families ka detection rate (in-design se kam aayega — **phir bhi report karo**, yahi credibility hai)
- GPTCache benchmark — **alag venv mein**, warna aapka environment downgrade ho jaayega:

```powershell
python -m venv .venv-gptcache
.\.venv-gptcache\Scripts\Activate.ps1
pip install gptcache
```

- Tenant isolation test
- Latency `percentile_cont` se apne hi decision log pe

---

# PHASE 10 — Ship
### 94% → 100%

- `/stats` JSON endpoint
- **Dockerfile likho** (Claude Code se karwa lo) — build local pe nahi, GitHub Actions pe hoga. Aapke laptop pe Docker nahi chahiye, resume pe Docker aa jaayega
- CI: lint+test, docker build, eval-gate
- Deploy Fly/Render + Neon
- **README as a report** — teen ghante. Ye documentation nahi, yehi deliverable hai
- `numbers.md` se saare brackets bhar do

---

# Kya khud likhna hai, kya Claude Code se

**Claude Code ko de do** — koi nahi poochta, aur time lagta hai:
Dockerfile · CI YAML · Pydantic schemas · migration runner · matplotlib formatting · test fixtures · FastAPI boilerplate

**Khud likho** — interview mein 10 minute inhi pe jaayenge:
`constraints.py` · admission loop · `run_sweep.py` (khaas kar wo `np.where` transform) · `cost_model.py` · perturbation families aur held-out split

**Beech ka rasta jo achha chalta hai:** failing test khud likho → Claude Code ko test dekar bolo "make it pass" → diff padho → jo pasand na aaye badal do. Design decisions aapke, typing bachi.

---

# Ek honest baat

Claude Code coding fast kar dega, ye sach hai. Lekin is plan ke asli bottleneck coding nahi hain — 100 pairs haath se verify karna, ye judge karna ki P2 aasan toh nahi, cost ratio se operating point choose karna. In teeno mein agent se koi speed-up nahi milta, aur Phase 7 aur 8 inhi pe tike hain.

Toh **paanch din ka plan hi rakhna**, teen ka mat banana.

---

# Progress table

| Phase | Kya | % |
|---|---|---|
| 0 | Installs | 8 |
| 1 | Accounts | 12 |
| 2 | Folder + Git | 20 |
| 3 | Cache redirect + venv | 30 |
| 4 | Model + data | 38 |
| 5 | Pehla Claude Code prompt (miss path) | 52 |
| 6 | Eval script, fake data | 58 |
| 7 | **Asli data, pehla asli number** | 72 |
| 8 | **Constraint gate, curve shift** | 84 |
| 9 | Defence (GPTCache, held-out) | 94 |
| 10 | Deploy + README | 100 |

Phase 7 tak pahunch gaye = project ho gaya. Uske baad sab improvement hai.

---

# Agar kuch atak jaaye

| Problem | Kya hua | Fix |
|---|---|---|
| `python` se Store khulta hai | PATH tick reh gaya | Installer → Modify → Add to PATH |
| "running scripts is disabled" | PowerShell policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Packages C drive pe ja rahe hain | env var nahi laga | Nayi PowerShell kholo, `echo $env:PIP_CACHE_DIR` check karo |
| Embeddings string aa rahi hain | `register_vector` pool pe nahi laga | `create_pool(..., init=register_vector)` |
| ROC ulti dikh rahi hai | `<=>` distance hai, similarity nahi | `similarity = 1 - distance` |
| Saari similarity ~0.99 | normalize nahi ho raha | fastembed by default normalize karta hai — confirm karo |
| Neon slow / connect nahi ho raha | free tier sleep ho jaata hai | Pehli query jagayegi, ~500ms. Latency measure karte waqt pehle warm karo |
