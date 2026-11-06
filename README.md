# qwen3.5-ita-agent-mlx

Piccoli modelli Qwen3.5 addestrati come agenti italiani per strumenti MCP e terminale, interamente in
[MLX](https://github.com/ml-explore/mlx) su Apple Silicon. Qui c'è il codice che ha costruito e misurato i due
modelli pubblicati:

- [`fabriziosalmi/Qwen3.5-2B-ita-agent-MLX-8bit`](https://huggingface.co/fabriziosalmi/Qwen3.5-2B-ita-agent-MLX-8bit)
- [`fabriziosalmi/Qwen3.5-0.8B-ita-agent-MLX-8bit`](https://huggingface.co/fabriziosalmi/Qwen3.5-0.8B-ita-agent-MLX-8bit)

<p align="center"><img src="assets/demo.svg" alt="Tre task del test congelato: Qwen3.5-2B originale contro Qwen3.5-2B-ita-agent" width="100%"></p>

L'agente legge e scrive file, gestisce note e appuntamenti, calcola date e IVA con gli strumenti, legge
pagine web, chiede conferma prima di cancellare o modificare dati, e risponde in italiano anche in una
semplice chat. Non è un modello per programmare. Tutto (generazione dei dati, training, valutazione) è girato
su due Mac mini M4 da 16 GB: nessuna GPU a noleggio, e nessun output di modelli chiusi nei dati di training.

La demo qui sopra mostra tre task veri del test congelato, eseguiti dal Qwen3.5-2B originale (senza harness,
come si usa di solito) e dal 2B addestrato con il suo harness; la genera `scripts/demo.py` leggendo i file di
`results/release/`, e ogni verdetto è verificato sui dati della corsa.

Cosa c'è nel repository:

- ambienti MCP deterministici, serviti come veri server MCP nello stesso processo (file, note, calendario,
  web, terminale, strumenti italiani per date e tasse, e due server usati solo nel test);
- un loop dell'agente con un harness leggero attorno al modello, e la valutazione di ogni traiettoria con
  controlli sullo stato degli ambienti e sulla risposta;
- set di valutazione congelati, e i risultati per task dei modelli pubblicati;
- mondi generati e famiglie di task che producono conversazioni di training verificate, per rejection
  sampling da un teacher Qwen3.5, con i filtri applicati;
- fine-tuning LoRA di Qwen3.5 in MLX, su un Mac o su due collegati come ring MLX.

## Risultati

Decodifica greedy, modelli MLX a 8 bit, thinking spento. Un task è superato solo se lo stato degli ambienti
e la risposta passano i loro controlli. Ogni numero qui sotto è calcolato dai file in `results/release/`.
"Harness" è l'insieme di flag usato al rilascio, `intent+append+lint+replace+noedit+gate+hide+cal2` (vedi
[Harness dell'agente](#harness-dellagente)); "nessuno" è il normale uso di mlx-lm con un system prompt.

| Modello | test, harness | test, nessuno | dev, harness | dev, nessuno | fresh5, harness |
|---|---|---|---|---|---|
| Qwen3.5-0.8B originale | 32/112 | 26/112 | 19/88 | 12/88 | 33/296 |
| Qwen3.5-0.8B-ita-agent | 49/112 | 41/112 | 54/88 | 38/88 | 204/296 |
| Qwen3.5-2B originale | 48/112 | 58/112 | 38/88 | 37/88 | 96/296 |
| Qwen3.5-2B-ita-agent | 66/112 | 63/112 | 65/88 | 55/88 | 233/296 |
| Qwen3.5-4B originale | 72/112 | | | | |

Test dei segni appaiati sugli stessi task, con l'harness:

| Confronto | Set | Vinti / persi | p |
|---|---|---|---|
| 2B-ita-agent contro 2B originale | test | +27 / -9 | 0,004 |
| 0.8B-ita-agent contro 0.8B originale | test | +24 / -7 | 0,003 |
| 2B-ita-agent contro 0.8B-ita-agent | test | +23 / -6 | 0,002 |
| 2B-ita-agent contro 4B originale | test | +17 / -23 | 0,43 |
| 2B-ita-agent contro 2B originale | fresh5 | +142 / -5 | < 0,001 |
| 0.8B-ita-agent contro 0.8B originale | fresh5 | +178 / -7 | < 0,001 |

Come leggerli:

- **test** (112 task in `eval/test/`) è il numero principale: scritto prima del training e letto una sola
  volta, dopo che ogni scelta era fatta. 45 dei suoi task usano due server che il modello non ha mai visto in
  training (certificati TLS e git).
- **dev** (88 task scritti a mano in `eval/dev/`, 10 categorie) ha guidato l'harness e i dati di training,
  quindi sopravvaluta i modelli addestrati.
- **fresh5** (296 task in `results/fresh5/`) viene dai generatori di task con seed mai usati in training:
  misura la distribuzione addestrata, non il trasferimento.
- Il 2B originale va meglio senza harness (58) che dentro l'harness (48): l'harness si basa su etichette dei
  turni che i modelli addestrati hanno imparato a produrre. Contro il 2B originale usato normalmente, il 2B
  addestrato con il suo harness guadagna poco sul totale dei task del test (66 contro 58). Il guadagno è
  nella sicurezza: supera 9 degli 11 task di sicurezza del test (chiede conferma o rifiuta prima di
  un'azione rischiosa), contro 4 su 11 del 2B originale usato normalmente.
- Il 2B addestrato resta sotto il 4B originale sul test, anche se non in modo significativo.

Due giri di training successivi (non pubblicati) non hanno battuto il 2B pubblicato sul dev e hanno perso
conoscenza generale dell'italiano sui set d'esame ITALIC e INCLUDE (perdite appaiate significative): per
questo non sono stati adottati.

### Harness dell'agente

Il loop dell'agente è `qita/harness/runner.py`; l'harness è un insieme di flag uniti da `+`, passato con
`--harness`. I pezzi principali:

- `intent`: ogni turno dell'utente viene prima classificato dal modello stesso come `rispondi`, `strumenti`,
  `chiedi` o `conferma`, e poi eseguito con quell'intento e la sua regola;
- `gate`, `hide`: in un turno `chiedi` o `conferma`, gli strumenti che cancellano o modificano dati vengono
  nascosti e rifiutati;
- `append`, `replace`, `noedit`, `lint`: i file si modificano con `append_to_file` e `replace_in_file`
  invece di riscriverli interi, e i file JSON e Python letti o scritti ricevono un verdetto sulla sintassi;
- `cal2`: il contesto elenca il calendario da oggi per i sette giorni successivi.

`agent` indica l'insieme predefinito attuale (`AGENT` in `qita/harness/runner.py`), cresciuto dopo il
rilascio con altri flag (per esempio `unseen` e `toolcls`, per server i cui strumenti il modello non ha mai
visto). Per riprodurre i numeri del rilascio, passa esplicitamente l'insieme del rilascio, come nei comandi
qui sotto.

### Regole di comportamento

I dati di training e l'harness seguono un'unica regola, scritta in italiano in `qita/harness/intent.py`
(`INTENTS`) e in `qita/worlds/style.py` (`POLICY_HINT`): leggere i dati con gli strumenti e non inventarli
mai; agire quando l'utente nomina con precisione cosa cancellare o modificare; quando la richiesta è ampia o
generica ("tutte le foto"), elencare cosa verrebbe cancellato e chiedere prima conferma; chiedere un dato che
manca (un giorno, un'ora) invece di indovinarlo; se la cosa da cancellare non esiste, dirlo e non proporre di
cancellare altro. Risposte brevi, markdown leggero, niente emoji.

## Regole sui dati

- Il teacher è un modello Qwen3.5 di al massimo 4B parametri, a pesi aperti con licenza Apache-2.0. Nei dati
  di training non entra nessun output di GPT, Claude, Gemini o altri modelli chiusi, né alcun dataset con
  licenza non commerciale.
- Una conversazione viene tenuta solo se passa i controlli sugli ambienti; poi controlli sul testo, una
  blocklist contro gli item dei benchmark e le richieste di valutazione, e la deduplicazione girano su ogni
  riga tenuta.
- I task di valutazione non entrano mai nei dati di training né nei prompt che li generano. I server usati
  solo nel test (`certs`, `git`, ruolo `test` in `qita/envs/__init__.py`) non compaiono mai nei dati di
  training, nei prompt di generazione o nelle decisioni di tuning; `qita/tasks.py` li rifiuta fuori dal test.
- Ogni risposta attesa e ogni controllo sullo stato di un task è calcolato dal codice degli ambienti, non
  scritto a mano.

I dati di training generati non sono pubblicati. Le fonti dei seed per le chat senza strumenti, con le
licenze e le fonti escluse, sono in `data/sources.yaml`. I testi dei seed non sono inclusi (gli atti della
Gazzetta Ufficiale nominano persone reali e i loro indirizzi): `scripts/seeds.py` li scarica dalle revisioni
fissate dei dataset in `data/seeds/`.

## Per cominciare

Requisiti: macOS su Apple Silicon, [uv](https://docs.astral.sh/uv/), Python 3.12.

```bash
uv sync
uv run pytest
```

Nessun test carica un modello. Due servono un tokenizer e senza vengono saltati: i test delle maschere di
loss in `tests/test_train_data.py` vogliono `Qwen/Qwen3.5-0.8B` nella cache locale di Hugging Face (per
esempio dopo `hf download Qwen/Qwen3.5-0.8B`), e un test dei prompt di MMLU-ProX vuole la cartella di un
modello Qwen3.5 in `QITA_TOKENIZER`.

Il dev sul 2B pubblicato, con l'harness del rilascio:

```bash
uv run python -m qita.eval --split dev --model fabriziosalmi/Qwen3.5-2B-ita-agent-MLX-8bit \
    --harness intent+append+lint+replace+noedit+gate+hide+cal2 --out runs/dev_2b.jsonl
uv run python -m qita.report runs/dev_2b.jsonl
```

`--model` accetta un id di Hugging Face o una cartella MLX locale; `--adapter` aggiunge adapter LoRA. I set
di task generati in `results/*/*_tasks.jsonl` si eseguono con `scripts/fresh_eval.py run --tasks ...`.

Su macOS, `scripts/jailed.sh NOME -m <modulo> ...` esegue un modulo in una sandbox del kernel
(`scripts/jail.sb`): scrive solo sotto `runs/jail/NOME`, niente rete, e nella home legge solo il repository,
la cartella dei modelli e l'installazione di Python. `scripts/jail_probe.py` verifica che i tentativi di uscita
vengano negati.

## Generazione dei dati e training

1. Generare conversazioni verificate: il teacher fa l'assistente in mondi generati, e si tengono solo le
   corse che passano i controlli (`qita/worlds/sample.py`; famiglie di task in `qita/worlds/families*.py`).

   ```bash
   uv run python -m qita.worlds.sample --teacher Qwen/Qwen3.5-4B --start 0 --n 20 --out data/gen/smoke
   ```

2. Filtrare: `scripts/store_filter.py` (famiglie dello store), `scripts/textcheck.py` (parole inglesi e
   grammatica; il controllo grammaticale richiede CLM, un correttore italiano separato e non pubblicato) e
   `scripts/blocklist.py` (sovrapposizioni con benchmark e valutazioni, duplicati).
3. Convertire nel formato dell'harness, con gli esempi di classificazione dei turni
   (`qita/worlds/harness_data.py`).
4. Addestrare con LoRA (`qita/train/sft.py`): loss solo sui token del modello, il chat template del modello e
   il formato di tool call `qwen3_coder`, gated delta rule a blocchi per Qwen3.5 (`qita/train/gdn_chunk.py`).

   ```bash
   uv run python -m qita.train.sft --model <cartella del modello MLX> --data a.kept.jsonl,a.cls.jsonl \
       --out runs/sft --rank 16 --scale 20 --lr 2e-5 --warmup 30 --epochs 1 --max-len 3072
   ```

I modelli pubblicati sono stati addestrati su due Mac collegati come un unico ring MLX (`scripts/ring.sh`,
host indicati con `QITA_RANK0` e `QITA_RANK1`), con i gradienti mediati a ogni passo; `--resume N` riprende
una corsa interrotta dal suo checkpoint. `scripts/remote.sh` sincronizza il repository su un altro Mac e lì
esegue i comandi. Gli adapter addestrati sono stati fusi nella base a 8 bit con `mlx_lm fuse`.

`scripts/check_patch.py`, `scripts/check_loss.py` e `scripts/check_cache.py` verificano su un modello vero
che il percorso di training a blocchi, la loss mascherata e la cache del prompt danno gli stessi valori di
mlx-lm.

## Struttura

```
qita/envs/       server MCP deterministici e i loro controlli sullo stato
qita/harness/    backend del modello (mlx-lm), loop dell'agente, flag dell'harness, valutazione, controllo dell'italiano
qita/worlds/     mondi generati, famiglie di task, rejection sampling, filtri dei dati
qita/train/      fine-tuning LoRA in MLX
qita/bench/      una scala di task sui file per iterare sull'harness
qita/eval.py     esegue uno split; qita/report.py e qita/compare.py riassumono i file dei risultati
qita/mcq.py      ITALIC e INCLUDE (domande d'esame in italiano), valutati per verosimiglianza della lettera
qita/mmlu_prox.py  MMLU-ProX-Lite in italiano, come il task mmlu_prox_lite_it di lm-eval
eval/            split dev e test, richieste di dogfood, MMLU-ProX-Lite (italiano)
fixtures/        la cartella home e le pagine web servite dagli ambienti
results/         set di task generati congelati e risultati per task dei modelli pubblicati
data/            le fonti dei seed per le chat senza strumenti (data/sources.yaml)
scripts/         filtri, sonde, verifiche, sandbox, esecuzione remota e ring, la demo
assets/          la demo animata (scripts/demo.py)
tests/           test unitari, ogni nuovo controllo con il suo break-test
```

## Test congelato

Il test è stato congelato prima di ogni training e letto una sola volta, dopo che ogni scelta era fatta. I
suoi file sono pubblicati senza modifiche; la loro prima riga di commento cita il registro del progetto, che
non è pubblicato.

| File | SHA-256 |
|---|---|
| `eval/test/train_envs.yaml` | `980b10d41fc4f64f06a0d5e62db60b37bf2d6b11fc6b30a69b50eb7d083852d4` |
| `eval/test/held_envs.yaml` | `4212be96057e60828e002fd7d78693ecba14318ed1275f6b0f7ad6e9cce653be` |
| `eval/test/multi.yaml` | `9006f815b7de80ccc91f01fc9ffabcbef0edfff7a29c78fad72d6421468376f6` |

## Limiti

- Misurato su ambienti simulati.
- Le aree più deboli del 2B pubblicato sul test: catene di chiamate dipendenti, chiarire una richiesta
  ambigua, riprendersi dopo un errore di uno strumento.
- Senza harness i modelli addestrati chiedono conferma prima di un'azione distruttiva meno spesso: nel tuo
  client, tieni gli strumenti distruttivi dietro una conferma.
- Solo italiano. Il thinking era spento in training e in valutazione.

## Licenza

Apache-2.0 (`LICENSE`), come i modelli base Qwen3.5 e il teacher. `eval/mmlu_prox_lite_it/` è una copia di
`li-lab/MMLU-ProX-Lite` (MIT, vedi il suo README). I testi dei seed scaricati da `scripts/seeds.py` mantengono
le licenze delle loro fonti, elencate in `data/sources.yaml`.
