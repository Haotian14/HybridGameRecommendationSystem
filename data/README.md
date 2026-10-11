# Data

The data is not stored in this repository: the sources are 2.4 GB. `hgrs prepare` rebuilds
exactly the files the project used from two public datasets, and verifies the result.

```bash
# 1. Download recommendations.csv (Kaggle, free account) into data/source/
# 2. Then run:
hgrs prepare      # also downloads the Steam Games file from Hugging Face; ~20 s after downloads
```

Expected output:

```
  interactions   950,355  ok
  users            8,225  ok
  games            3,654  ok
  fingerprint  c2201fd42a787206…  ok
```

## Sources

| | Game Recommendations on Steam | Steam Games Dataset |
|---|---|---|
| What | 41,154,794 user reviews: user, game, hours played, recommended or not | metadata for 111,452 games: description, genres, tags, developer, price, … |
| Author | Anton Kozyriev | Martin Bustos (FronkonGames) |
| Where | [Kaggle](https://www.kaggle.com/datasets/antonkozyriev/game-recommendations-on-steam) | [Hugging Face](https://huggingface.co/datasets/FronkonGames/steam-games-dataset) (mirror of [Kaggle](https://www.kaggle.com/datasets/fronkongames/steam-games-dataset)) |
| Version used | v28, 2024-08-14 | revision [`6ba5504`](https://huggingface.co/datasets/FronkonGames/steam-games-dataset/tree/6ba55046b084aba5cbd62c64ce38fa7182058e42), 2025-04-26 |
| File | `recommendations.csv` (2,023,223,067 bytes) | `games.csv` (335,236,382 bytes) |
| sha256 | `18731b9e253b63ad8642dde48c212620621c33080d7609fa092a308caaffbb34` | `c621746e5c6d60bd193d7748ed4b32d06c6e25d1bad81fd4615d24f74b7a2ce5` |
| Licence | CC0 1.0 (public domain) | MIT, as listed on Kaggle and Hugging Face |
| How to get it | log in to Kaggle, download `recommendations.csv` from the dataset page, unzip it to `data/source/recommendations.csv` | downloaded automatically by `hgrs prepare` (public, no login) |

**Why these versions.** The Steam Games Dataset is updated regularly. The revision above is the
one that was current when the project was built (July 2025), and its metadata matches the
project's data field for field. Both files are verified by checksum before use.

## Pipeline

`hgrs prepare` follows the original project's steps (`src/hgrs/prepare.py`):

1. **Repair the Steam Games header.** It fuses two column names into `DiscountDLC count`, so every
   name after `Price` sits one column to the left of its data. The project's `fixed_file.csv`
   repaired it the same way.
2. **Attach genres to reviews.** Each review gets its game's `Genres`; reviews of games with no
   genre are dropped: 41,154,794 → 40,954,828 reviews (the project's `user_game_with_genre.csv`).
3. **Keep active users and popular games.** Users with ≥ 100 reviews and games with ≥ 1,000
   reviews, both counted *before* filtering (so a kept user can have fewer than 100 reviews left
   among kept games).
4. **Write the outputs** to `data/raw/`.

## Outputs

| File | Rows | Columns |
|---|---:|---|
| `filtered_games.csv` | 950,355 | `app_id, is_recommended, hours, user_id, genre` |
| `games.csv` | 3,654 | `app_id, Name, Price, About the game, Developers, Publishers, Categories, genre, Tags` |

The fingerprint is a sha256 of the interactions sorted and written in a fixed format, so it does
not depend on row order or on how a pandas version prints floats. It matches the project's
original `filtered_games.csv` exactly.

## Test profiles

[`profiles/user_sport.csv`](profiles/user_sport.csv) and
[`profiles/user_call_of_duty.csv`](profiles/user_call_of_duty.csv) are two hand-built profiles
that the project added to the data to inspect recommendations: a sports and racing fan (user
`22222222`, 18 games) and a Call of Duty / FPS fan (user `11111111`, 24 games). They are not
derived from the sources, so they are kept in the repository. Their game metadata columns come
from the Steam Games Dataset.
