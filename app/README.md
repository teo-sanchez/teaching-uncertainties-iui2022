# Uncertainty Guidance

> A [Marcelle](https://marcelle.netlify.app) Application

---

## Services

- Nomenclature :
  [nom_de_la_tâche]-[id_du_participant]-[condition].db

- Interface des messages :

```ts
export interface Message<T> {
  content: {
    user: {
      id: Id;
    };
    task: {
      name: string;
      condition: string;
      iteration: number;
    };
    data: T;
  };
  // those are added by feathers
  id?: Id;
  createdAt?: Date;
  updatedAt?: Date;
}
```

## Problèmes connus, à faire

- réinitialiser valeurs incertitudes test (0.5)
- Changer de page lors des test de précision et d'incertitude fait avancer l'indice des images (on enregistre automatiquement + on incrémente.)

## Notes

- **Recompiler en `yarn dev-client` après tout changement structurel de fichiers JSON.**
- Citer _Scikit-learn_ si on l'utilise : https://scikit-learn.org/stable/about.html#citing-scikit-learn
