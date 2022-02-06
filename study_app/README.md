# Uncertainty Guidance

> A [Marcelle](https://marcelle.netlify.app) Application

---

## Installation

```setup
npm install
```
or
```setup
yarn install
```

## Running
```setup
npm run dev
```
or
```setup
yarn dev
```

## Services

- Nomenclature :
  [task-name]-[participan-id]-[condition].db

- Messages interface :

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
## Notes

- **Recompil with `yarn dev-client` after all structural changes in JSON files.**