# Présentation : Minecraft Mob Vision

Slides reveal.js (Livrable 2). Autonome : reveal.js est chargé depuis un CDN, les figures sont dans `assets/`.

## Lancer

Ouvrir `index.html` dans un navigateur. Ou servir en local (mieux pour le routage `hash`) :

```powershell
# depuis la racine du repo
uv run python -m http.server 8000 --directory presentation
# puis ouvrir http://localhost:8000
```

## Contrôles

- `→` / `Espace` : suivant, `←` : précédent
- `Échap` : vue d'ensemble
- `S` : notes du présentateur
- `F` : plein écran

## Régénérer les figures

Tous les graphiques de `assets/` sont produits par `notebook.ipynb`. Relancer le notebook pour les rafraîchir :

```powershell
uv run python build_notebook.py
uv run jupyter nbconvert --to notebook --execute --inplace notebook.ipynb
```
