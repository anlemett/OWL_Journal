import sys

def get_grid_search_params(model, best_params):
    if model == 'KNN':
        params = {
            'classifier__n_neighbors': [best_params['classifier__n_neighbors'] - 2, best_params['classifier__n_neighbors'], best_params['classifier__n_neighbors'] + 2],  # Nearby values for n_neighbors
            'classifier__weights': [best_params['classifier__weights'], 'uniform' if best_params['classifier__weights'] == 'distance' else 'distance'],  # Toggle weights
            'classifier__algorithm': [best_params['classifier__algorithm'], 'auto'],  # Keep best algorithm with an additional option
            'classifier__p': [best_params['classifier__p'], 1 if best_params['classifier__p'] == 2 else 2],  # Toggle p between 1 and 2
            }
    elif model == 'SVC':
            params = {
                'classifier__C': [best_params['classifier__C'] * 0.1, best_params['classifier__C'], best_params['classifier__C'] * 10],
                'classifier__kernel': [best_params['classifier__kernel']],  # Keep the same kernel as found in Random Search
                #'classifier__gamma': [best_params['classifier__gamma'] * 0.1, best_params['classifier__gamma'], best_params['classifier__gamma'] * 10]
                'classifier__gamma': (
                        [best_params['classifier__gamma'] * 0.1,
                         best_params['classifier__gamma'],
                         best_params['classifier__gamma'] * 10
                        ]
                if isinstance(best_params['classifier__gamma'], (int, float))
                else [best_params['classifier__gamma']]  # Preserve 'scale' or 'auto'
                )
                }
    elif model == "HGBC":
            print(best_params)
            params = {
                'classifier__learning_rate':
                    [best_params['classifier__learning_rate'] * 0.5,
                     best_params['classifier__learning_rate'],
                     best_params['classifier__learning_rate'] * 1.5],
                'classifier__max_iter':
                    [best_params['classifier__max_iter'] - 50, 
                     best_params['classifier__max_iter'],
                     best_params['classifier__max_iter'] + 50],
                'classifier__max_depth':
                    [best_params['classifier__max_depth']]
                    if best_params['classifier__max_depth'] is None else
                       [best_params['classifier__max_depth'] - 5,
                       best_params['classifier__max_depth'],
                       best_params['classifier__max_depth'] + 5], 
                #'classifier__min_samples_leaf': [best_params['classifier__min_samples_leaf'] - 5, best_params['classifier__min_samples_leaf'], best_params['classifier__min_samples_leaf'] + 5],
                'classifier__min_samples_leaf': [best_params['classifier__min_samples_leaf']], # keep the same
                #'classifier__l2_regularization': [best_params['classifier__l2_regularization'] * 0.5, best_params['classifier__l2_regularization'], best_params['classifier__l2_regularization'] * 1.5]
                'classifier__l2_regularization': [best_params['classifier__l2_regularization']] # keep the same
                }

    elif model == "NC":
        params = {
            'classifier__metric': [best_params['classifier__metric']],  # Keep the best metric
            'classifier__shrink_threshold': [
                max(0, best_params['classifier__shrink_threshold'] - 0.1),
                best_params['classifier__shrink_threshold'],
                min(1, best_params['classifier__shrink_threshold'] + 0.1),
                ],
            }
    elif model == "RNC":
        params = {
            'classifier__radius': [best_params['classifier__radius'] * 0.5, best_params['classifier__radius'], best_params['classifier__radius'] * 1.5],  # Nearby values for radius
            'classifier__weights': [best_params['classifier__weights'], 'uniform' if best_params['classifier__weights'] == 'distance' else 'distance'],  # Toggle weights
            'classifier__algorithm': [best_params['classifier__algorithm'], 'auto'],  # Keep best algorithm with an additional option
            'classifier__leaf_size': [best_params['classifier__leaf_size'] - 10, best_params['classifier__leaf_size'], best_params['classifier__leaf_size'] + 10]
            }
    elif model == "ETC":
        params = {
            'classifier__n_estimators': [best_params['classifier__n_estimators'] - 50, best_params['classifier__n_estimators'], best_params['classifier__n_estimators'] + 50],
            'classifier__max_features': [best_params['classifier__max_features']],  # Keep the best max features found by Random Search
            'classifier__max_depth': [best_params['classifier__max_depth'] - 5, best_params['classifier__max_depth'], best_params['classifier__max_depth'] + 5],
            'classifier__min_samples_split': [best_params['classifier__min_samples_split'] - 2, best_params['classifier__min_samples_split'], best_params['classifier__min_samples_split'] + 2],
            'classifier__min_samples_leaf': [best_params['classifier__min_samples_leaf'] - 2, best_params['classifier__min_samples_leaf'], best_params['classifier__min_samples_leaf'] + 2],
            }
    elif model == "MLPC":
        params = {
            'classifier__hidden_layer_sizes': [(best_params['classifier__hidden_layer_sizes'][0] - 25,), 
                           best_params['classifier__hidden_layer_sizes'], 
                           (best_params['classifier__hidden_layer_sizes'][0] + 25,)], 
            'classifier__activation': [best_params['classifier__activation']],  # Keep the best activation function found by Random Search
            'classifier__solver': [best_params['classifier__solver']],  # Keep the best solver found by Random Search
            'classifier__alpha': [best_params['classifier__alpha'] * 0.5, best_params['classifier__alpha'], best_params['classifier__alpha'] * 1.5],
            'classifier__learning_rate': [best_params['classifier__learning_rate']],  # Keep the best learning rate found by Random Search
            'classifier__learning_rate_init': [best_params['classifier__learning_rate_init'] * 0.5, 
                           best_params['classifier__learning_rate_init'], 
                           best_params['classifier__learning_rate_init'] * 1.5],
            }
    elif model == "LDA":
        params = {
            'classifier__solver': [best_params['classifier__solver']],  # Fix the solver to the best found
            'classifier__shrinkage': [
                max(0, best_params['classifier__shrinkage'] - 0.1),  # Reduce shrinkage slightly
                best_params['classifier__shrinkage'],
                min(1, best_params['classifier__shrinkage'] + 0.1)   # Increase shrinkage slightly
                ] if best_params['classifier__solver'] in ['lsqr', 'eigen'] else ['auto']  # 'auto' if shrinkage isn't used
            }
    elif model == "QDA":
        params = {
            'classifier__reg_param': [
                max(0, best_params['classifier__reg_param'] - 0.1),
                best_params['classifier__reg_param'],
                min(1, best_params['classifier__reg_param'] + 0.1),
                ],
            'classifier__tol': [
                max(1e-6, best_params['classifier__tol'] - 1e-5),
                best_params['classifier__tol'],
                best_params['classifier__tol'] + 1e-5,
                ],
            }
    elif model == 'RF':
            params = {
                'classifier__n_estimators': [best_params['classifier__n_estimators'] - 50, best_params['classifier__n_estimators'], best_params['classifier__n_estimators'] + 50],
                'classifier__max_depth': [best_params['classifier__max_depth'] - 10, best_params['max_depth'], best_params['classifier__max_depth'] + 10] if best_params['classifier__max_depth'] else [None],
                'classifier__max_features': [best_params['classifier__max_features']],
                'classifier__min_samples_leaf': [best_params['classifier__min_samples_leaf'] - 1, best_params['classifier__min_samples_leaf'], best_params['classifier__min_samples_leaf'] + 1],
                #'classifier__min_samples_split': [best_params['min_samples_split'] - 1, best_params['min_samples_split'], best_params['min_samples_split'] + 1]
                'classifier__min_samples_split': [best_params['classifier__min_samples_split']]
                }
    elif model == "ABC":
        params = {
            'classifier__n_estimators': [
                max(50, best_params['classifier__n_estimators'] - 20), 
                best_params['classifier__n_estimators'], 
                best_params['classifier__n_estimators'] + 20
                ],
            'classifier__learning_rate': [
                max(0.01, best_params['classifier__learning_rate'] - 0.1), 
                best_params['classifier__learning_rate'], 
                best_params['classifier__learning_rate'] + 0.1
                ],
            'classifier__base_estimator__max_depth': [
                max(1, best_params['classifier__estimator__max_depth'] - 1), 
                best_params['classifier__estimator__max_depth'], 
                best_params['classifier__estimator__max_depth'] + 1
                ],
            'classifier__base_estimator__min_samples_split': [
                max(2, best_params['classifier__estimator__min_samples_split'] - 1), 
                best_params['classifier__estimator__min_samples_split'], 
                best_params['classifier__estimator__min_samples_split'] + 1
                ]
            }
    elif model == "BC":
        params = {
            'classifier__n_estimators': [
                max(10, best_params['classifier__n_estimators'] - 10), 
                best_params['classifier__n_estimators'], 
                best_params['classifier__n_estimators'] + 10
                ],
            
            'classifier__max_samples': [
                max(0.1, best_params['classifier__max_samples'] - 0.1), 
                best_params['classifier__max_samples'], 
                min(1.0, best_params['classifier__max_samples'] + 0.1)
                ],
            
            'classifier__max_features': [
                max(0.1, best_params['classifier__max_features'] - 0.1), 
                best_params['classifier__max_features'], 
                min(1.0, best_params['classifier__max_features'] + 0.1)
                ],
            
            'classifier__estimator__max_depth': [
                max(1, best_params['classifier__estimator__max_depth'] - 1), 
                best_params['classifier__estimator__max_depth'], 
                best_params['classifier__estimator__max_depth'] + 1
                ],
            
            'classifier__estimator__min_samples_split': [
                max(2, best_params['classifier__estimator__min_samples_split'] - 1), 
                best_params['classifier__estimator__min_samples_split'], 
                best_params['classifier__estimator__min_samples_split'] + 1
                ]
        }

    elif model == "GBC":
        params = {
            'classifier__n_estimators': [best_params['classifier__n_estimators'] - 50, best_params['classifier__n_estimators'], best_params['classifier__n_estimators'] + 50],
            'classifier__learning_rate': [best_params['classifier__learning_rate'] * 0.5, best_params['classifier__learning_rate'], best_params['classifier__learning_rate'] * 1.5],
            'classifier__max_depth': [best_params['classifier__max_depth'] - 1, best_params['classifier__max_depth'], best_params['classifier__max_depth'] + 1],
            'classifier__min_samples_split': [best_params['classifier__min_samples_split'] - 2, best_params['classifier__min_samples_split'], best_params['classifier__min_samples_split'] + 2],
            'classifier__min_samples_leaf': [best_params['classifier__min_samples_leaf'] - 1, best_params['classifier__min_samples_leaf'], best_params['classifier__min_samples_leaf'] + 1]
            }
    elif (model == "SC") or (model == "VC"):
        params = {
            'classifier__svc__C': [
                max(0.01, best_params['classifier__svc__C'] - 0.5),
                best_params['classifier__svc__C'],
                best_params['classifier__svc__C'] + 0.5,
                ],
            'classifier__svc__gamma': (
                [best_params['classifier__svc__gamma'] * 0.1,
                 best_params['classifier__svc__gamma'],
                 best_params['classifier__svc__gamma'] * 10
                 ]
                if isinstance(best_params['classifier__svc__gamma'], (int, float))
                else [best_params['classifier__svc__gamma']]  # Preserve 'scale' or 'auto'
                ),
            'classifier__knn__n_neighbors': [
                max(3, best_params['classifier__knn__n_neighbors'] - 2),
                best_params['classifier__knn__n_neighbors'],
                best_params['classifier__knn__n_neighbors'] + 2,
                ],
            'classifier__hgb__learning_rate': [
                max(0.01, best_params['classifier__hgb__learning_rate'] - 0.05),
                best_params['classifier__hgb__learning_rate'],
                best_params['classifier__hgb__learning_rate'] + 0.05,
                ],
            'classifier__hgb__max_iter': [
                max(50, best_params['classifier__hgb__max_iter'] - 20),
                best_params['classifier__hgb__max_iter'],
                best_params['classifier__hgb__max_iter'] + 20,
                ],
            }
    elif model == "LP":
        params = {
            'classifier__kernel': [best_params['classifier__kernel']],
            'classifier__gamma': [
                max(0.01, best_params['classifier__gamma'] - 0.1),
                best_params['classifier__gamma'],
                best_params['classifier__gamma'] + 0.1
                ] if 'classifier__gamma' in best_params else [None]
            }
    elif model == "LS":
        params = {
            'classifier__kernel': [best_params['classifier__kernel']],
            'classifier__gamma': [
                max(0.01, best_params['classifier__gamma'] - 0.1),
                best_params['classifier__gamma'],
                best_params['classifier__gamma'] + 0.1
                ] if 'classifier__gamma' in best_params else [None],
            'classifier__alpha': [
                max(0.01, best_params['classifier__alpha'] - 0.05),
                best_params['classifier__alpha'],
                best_params['classifier__alpha'] + 0.05
                ]
            }
    else:
        print("Model not supported")
        sys.exit(0)
    
    # Ensure parameter values are valid
    params = {
        key: [
            val for val in values if val is None or isinstance(val, (str)) or isinstance(val, (tuple)) or val > 0
            ]
        for key, values in params.items()
        }
    
    return params
   
