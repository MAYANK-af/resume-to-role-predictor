# 🌌 3D Word Constellation — Resume-to-Role Frontend

Interactive 3D web experience built with **React 19**, **Three.js**, **@react-three/fiber**, **@react-three/drei**, and **GSAP**.

- 🌐 **Live Demo**: [https://resume-to-role-predictor.onrender.com](https://resume-to-role-predictor.onrender.com)
- 🔗 **Main Repository**: [https://github.com/MAYANK-af/resume-to-role-predictor](https://github.com/MAYANK-af/resume-to-role-predictor)

---

## 🛠️ Development & Build

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the Oxlint configuration

If you are developing a production application, we recommend enabling type-aware lint rules by installing `oxlint-tsgolint` and editing `.oxlintrc.json`:

```json
{
  "$schema": "./node_modules/oxlint/configuration_schema.json",
  "plugins": ["react", "typescript", "oxc"],
  "options": {
    "typeAware": true
  },
  "rules": {
    "react/rules-of-hooks": "error",
    "react/only-export-components": ["warn", { "allowConstantExport": true }]
  }
}
```

See the [Oxlint rules documentation](https://oxc.rs/docs/guide/usage/linter/rules) for the full list of rules and categories.
