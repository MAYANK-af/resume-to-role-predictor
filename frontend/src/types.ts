export type RoleName = 'Backend' | 'AI/ML' | 'Web Dev' | 'Analyst' | 'Other Tech';

export interface WordWeight {
  word: string;
  weight: number;
  tfidf?: number;
  coef?: number;
}

export interface PredictResponse {
  predicted_role: RoleName;
  nb_predicted_role: RoleName;
  target_role: RoleName;
  fit_score: number;
  lr_prob_fit_score: number;
  probabilities: Record<RoleName, number>;
  top_words: WordWeight[];
  target_words: WordWeight[];
  cleaned_tokens: string[];
}

export interface ModelComparison {
  Model: string;
  Accuracy: number;
  'Macro Precision': number;
  'Macro Recall': number;
  'Macro F1': number;
}

export interface RegressionComparison {
  'Target Role': string;
  'Ridge MAE': number;
  'Ridge R2': number;
  'LR Prob MAE': number;
  'LR Prob R2': number;
}

export interface TopWordMetadata {
  token: string;
  display: string;
  weight: number;
}

export interface MetadataResponse {
  roles: RoleName[];
  top_coefficients: Record<RoleName, TopWordMetadata[]>;
  comparison_metrics: ModelComparison[];
  regression_comparison: RegressionComparison[];
}

export type SceneState = 'landing' | 'typing' | 'analyzing' | 'result' | 'error';

export interface ConstellationWordNode {
  id: string;
  text: string;
  pos: [number, number, number];
  targetPos: [number, number, number];
  weight: number;
  isInfluential: boolean;
  role?: RoleName;
}

export interface RoleAnchorInfo {
  role: RoleName;
  name: string;
  position: [number, number, number];
  geometryType: 'box' | 'icosahedron' | 'octahedron' | 'dodecahedron' | 'torus';
  description: string;
}
