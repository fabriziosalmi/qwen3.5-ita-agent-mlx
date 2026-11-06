"""Slot values for generated worlds: plain data (names, places, goods, amounts), no prose."""

FIRST = ["Alessandro", "Alice", "Andrea", "Anna", "Beatrice", "Carlo", "Caterina", "Chiara", "Cristina",
         "Daniele", "Davide", "Elena", "Elisa", "Emma", "Enrico", "Federica", "Federico", "Filippo",
         "Francesca", "Francesco", "Gabriele", "Giacomo", "Gianni", "Giorgia", "Giorgio", "Giovanni",
         "Irene", "Laura", "Leonardo", "Lorenzo", "Luca", "Lucia", "Marta", "Martina", "Matilde",
         "Matteo", "Michele", "Nicola", "Paola", "Pietro", "Riccardo", "Roberta", "Roberto", "Silvia",
         "Simone", "Stefano", "Tommaso", "Valentina", "Valerio", "Vittoria"]
LAST = ["Amato", "Barbieri", "Bassi", "Benedetti", "Bernardi", "Bruno", "Caputo", "Cattaneo", "Colombo",
        "Conti", "Costa", "D'Angelo", "De Luca", "Esposito", "Fabbri", "Ferrara", "Ferraro", "Fontana",
        "Galli", "Gallo", "Gentile", "Giordano", "Greco", "Leone", "Lombardi", "Longo", "Mancini",
        "Marchetti", "Mariani", "Marini", "Martini", "Mazza", "Monti", "Moretti", "Morelli", "Neri",
        "Pellegrini", "Rinaldi", "Rizzo", "Romano", "Russo", "Santoro", "Serra", "Silvestri", "Testa",
        "Valentini", "Villa", "Vitale"]
CITIES = ["Torino", "Milano", "Bergamo", "Brescia", "Verona", "Padova", "Venezia", "Trieste", "Trento",
          "Bologna", "Parma", "Modena", "Firenze", "Pisa", "Livorno", "Perugia", "Ancona", "Pescara",
          "Roma", "Latina", "Napoli", "Salerno", "Bari", "Lecce", "Taranto", "Cosenza", "Palermo",
          "Catania", "Messina", "Cagliari", "Sassari", "Savona", "La Spezia", "Como", "Varese", "Novara"]
STREETS = ["via Roma", "via Garibaldi", "via Mazzini", "corso Vittorio Emanuele", "via Dante", "via Cavour",
           "piazza della Libertà", "via Verdi", "viale Europa", "via dei Mille", "via Manzoni", "corso Italia"]
GROCERIES = ["pane", "latte", "burro", "farina", "zucchero", "caffè", "tè", "riso", "pasta", "olio",
             "aceto", "sale", "pomodori", "zucchine", "melanzane", "cipolle", "patate", "carote", "mele",
             "pere", "arance", "limoni", "banane", "yogurt", "mozzarella", "parmigiano", "prosciutto",
             "uova", "tonno", "ceci", "lenticchie", "biscotti", "acqua frizzante", "detersivo",
             "carta igienica", "sapone", "spugne", "sacchi della spazzatura"]
SUPPLIERS = ["Tipografia", "Studio Tecnico", "Officina", "Idraulica", "Elettrica", "Ferramenta",
             "Cartoleria", "Studio Legale", "Agenzia", "Falegnameria", "Pasticceria", "Lavanderia"]
COMPANY_SUFFIX = ["s.r.l.", "s.n.c.", "s.a.s.", "di {last}"]
SERVICES = ["manutenzione caldaia", "stampa volantini", "consulenza fiscale", "riparazione infissi",
            "revisione impianto elettrico", "fornitura cancelleria", "pulizia uffici", "traduzione documenti",
            "noleggio attrezzature", "assistenza informatica"]
VAT_RATES = [4, 5, 10, 22, 22, 22]
EVENT_KINDS = [("Dentista", 60), ("Palestra", 60), ("Visita medica", 30), ("Riunione", 90), ("Call con {name}", 30),
               ("Cena con {name}", 120), ("Corso di inglese", 90), ("Parrucchiere", 60), ("Commercialista", 60),
               ("Colloquio", 45), ("Assemblea condominiale", 120), ("Revisione auto", 60), ("Partita di calcetto", 90)]
NOTE_TOPICS = [("Pagare il bollo dell'auto entro il {date}", ["auto", "scadenze"]),
               ("Codice cliente della luce: {code}", ["casa", "bollette"]),
               ("Chiamare {name} per {thing}", ["telefonate"]),
               ("Comprare il regalo per {name}", ["regali"]),
               ("Prenotare il tagliando dell'auto", ["auto"]),
               ("Restituire il libro in biblioteca entro il {date}", ["scadenze"]),
               ("Password del wifi ospiti nel cassetto della scrivania", ["casa"]),
               ("Numero della pratica in comune: {code}", ["pratiche"])]
THINGS = ["il preventivo", "la cena di sabato", "le chiavi della cantina", "il trasloco", "la riunione", "le foto"]
FOLDERS = {"docs": ["documenti", "Documenti", "docs", "carte"], "work": ["lavoro", "ufficio", "progetti"],
           "down": ["scaricati", "Download", "download"], "photos": ["foto", "Immagini", "immagini"],
           "notes": ["appunti", "note", "memo"], "logs": ["log", "logs", "registri"]}
SITE_KINDS = ["comune", "biblioteca", "farmacia", "palestra", "museo", "piscina", "ufficio postale", "asl"]
WEEKDAYS = ["lunedì", "martedì", "mercoledì", "giovedì", "venerdì", "sabato", "domenica"]
MONTHS = ["gennaio", "febbraio", "marzo", "aprile", "maggio", "giugno", "luglio", "agosto", "settembre",
          "ottobre", "novembre", "dicembre"]

# world v2
EXPENSE_CATS = {"spesa": ["supermercato", "mercato", "panetteria"], "bollette": ["luce", "gas", "acqua", "internet"],
                "trasporti": ["benzina", "treno", "autobus", "parcheggio"], "farmacia": ["farmaci", "visita"],
                "ristorante": ["pizzeria", "trattoria", "bar"], "casa": ["ferramenta", "detersivi", "idraulico"]}
TODO = ["rinnovare la carta d'identità", "portare l'auto dal meccanico", "pagare la rata del condominio",
        "sistemare la cantina", "chiamare l'assicurazione", "comprare le lampadine", "prenotare le vacanze",
        "restituire il trapano a {name}", "scrivere a {name}", "fare il cambio degli armadi",
        "aggiornare il curriculum", "iscriversi in palestra"]
DOWNLOADS = ["bolletta-luce.pdf", "modulo-iscrizione.pdf", "manuale-lavatrice.pdf", "biglietto-treno.pdf",
             "contratto-affitto.pdf", "ricevuta-bonifico.pdf", "foto-documento.zip", "estratto-conto.csv",
             "programma-corso.pdf", "orario-scuola.pdf"]
APP_EVENTS = {"INFO": ["richiesta servita", "utente autenticato", "cache aggiornata", "job notturno completato"],
              "WARN": ["risposta lenta dal database", "quota disco oltre l'80%", "tentativo di accesso ripetuto"],
              "ERROR": ["connessione al database persa", "timeout del servizio pagamenti", "file di configurazione illeggibile"]}
NEWS = ["Lavori in corso sulla linea del tram", "Nuovo orario estivo della biblioteca civica",
        "Festa di quartiere in piazza", "Chiusura straordinaria degli uffici per sciopero"]
