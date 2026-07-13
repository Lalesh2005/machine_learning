# PageRank Activity — Explore the Endorsement Network You Built

> **This is purely exploratory — not graded, nothing to submit.** It's a chance to play with a real network *you* helped create and to learn how PageRank works from the inside. Do as much or as little as you enjoy.

Attached: **`endorsement_network.csv`** — the network *you* created. Each row is one endorsement:

```
endorser, endorsed, weight
```

It means **"endorser found endorsed impressive"** after speaking with them. It is a **directed** graph: 500 people, ~1,762 endorsements. Names only (no emails). Where two people share a name they are numbered (e.g. *Jatin Kumar 1*, *Jatin Kumar 2*).

Use any tool you like — Python (`networkx`, `pandas`), R, a spreadsheet, or write the algorithm from scratch. The questions go from easy to deep — wander through whichever ones spark your curiosity.

---

### Part A — Get to know the data
1. Load the file. How many unique people are there? How many endorsements? Why is this a *directed* graph and not an undirected one?
2. **Out-degree:** who endorsed the most people? **In-degree:** who was endorsed by the most people? Are the two top-10 lists the same? What does that tell you?
3. How many people were endorsed by *nobody* (in-degree 0)? How many endorsed *nobody*?

### Part B — PageRank (the main event)
4. Implement or run **PageRank** on the network. List your **top 10**.
5. Compare your PageRank top-10 to the raw **in-degree** (most-endorsed) top-10. Where do they **disagree**, and **why**? Find one person who is high on raw count but low on PageRank, and one who is the opposite. Explain each.
6. Why does **endorsing many people NOT raise your own PageRank**? What *does* raise it?
7. Change the **damping factor** (try 0.5, 0.85, 0.95). How stable is the top 10? What is the damping factor actually doing?
8. PageRank was invented to rank **web pages** by links. Write 2–3 sentences mapping that idea onto *this* network: what is a "page," what is a "link," and what does a high score mean here?

### Part C — Network structure
9. **Reciprocity:** what fraction of endorsements are mutual (A→B *and* B→A)? What does a high number say about how the activity actually happened?
10. **Communities:** find the connected components / clusters. Is there one big group or many? Can you guess what the clusters correspond to (colleges, teams, friend groups)?
11. **Triangles & clustering:** count triangles. Socially, what is a triangle in this network? What does a high clustering coefficient mean?
12. **Bridges:** compute **betweenness centrality**. Who are the people connecting otherwise-separate groups? Are they the same as the PageRank stars, or different people?

### Part D — Think critically
13. Some people filled the form from an **alternate email**, so the same person could appear twice. How would undetected duplicates distort PageRank? How would you detect them?
14. How could someone **game** their own PageRank (self-endorsement, mutual-endorsement rings, etc.)? How would you detect or prevent it? Can you find any suspicious patterns in the data?
15. Is PageRank a **fair** measure of who is "impressive"? What does it reward, and what does it miss? Suggest one improvement.

### Part E — Reflect
16. If this network were used to make a *real* decision (a stipend, a team lead, a recommendation), what would you do with it — and what could go wrong?

---

**No submission, no grade.** There is no single right answer for Parts C–E — the goal is to think, tinker, and enjoy it. If you discover something interesting, share it on the forum so others can see it. Have fun, and feel free to go well beyond these questions.
