let Enums = ./enums.dhall
let ConfigTypes = ./config_types.dhall
let scratchPath = (./utils.dhall).scratchPath

let SavingConfig =
          ConfigTypes.SavingConfig_t
      /\  { default =
            { algorithm = "ERM"
            , phase = Enums.Phase.train
            , save_folder = "${scratchPath}/results/"
            }
          }

let TrainingDatasetConfig =
          ConfigTypes.TrainingDatasetConfig_t
      /\  { default =
            { data_dir = "${scratchPath}/data/"
            , n_classes = 182
            , train_dataset = Enums.Dataset.iwildcam-train
            , size = [ 224, 224 ]
            , n_channels = 3
            , train_transforms = [] : List Enums.Transform
            , eval_transforms = [ Enums.Transform.wilds_default_normalization ]
            , min_performance = 0.0
            , max_performance = 1.0
            }
          }

let ResourceConfig =
          ConfigTypes.ResourceConfig_t
      /\  { default =
            { max_num_workers = 32, accelerator = Enums.Accelerator.gpu, devices = 1 }
          }

let TrainingParamConfig =
          ConfigTypes.TrainingParamConfig_t
      /\  { default =
            { max_epochs = 12
            , lr = 0.001
            , batch_size = 24
            , optimizer = Enums.Optim.adam
            , model_source = Enums.ModelLibs.torchvision
            , model = Enums.Models.resnet50
            , label_smoothing = 0.0
            , pretrained_weights = Enums.PretrainedWeights.DEFAULT
            , freeze_weights = False
            , unfreeze_k_layers = 0
            , calibration_method = None Enums.CalibrationMethod
            , adversarial_training_method = None Enums.AdversarialAttack
            , adversarial_training_eps = None < Val : Double | Expr : Text >
            }
          }

let HRESetup =
          ConfigTypes.HRESetup_t
      /\  { default =
            { val_id_dataset = Enums.Dataset.iwildcam-id_val
            , val_ds_datasets =
              [ Enums.Dataset.iwildcam-val, Enums.Dataset.iwildcam-id_val-corruption1_val ]
            , val_ood_datasets =
              [ Enums.Dataset.gaussian_noise
              , Enums.Dataset.fmow-id_val
              , Enums.Dataset.rxrx1-id_val
              , Enums.Dataset.camelyon17-id_val
              ]
            , test_id_dataset = Enums.Dataset.iwildcam-id_test
            , test_ds_datasets =
              [ Enums.Dataset.iwildcam-test
              , Enums.Dataset.iwildcam-id_test-corruption1_test
              ]
            , test_ood_datasets =
              [ Enums.Dataset.gaussian_noise
              , Enums.Dataset.fmow-id_test
              , Enums.Dataset.rxrx1-id_test
              , Enums.Dataset.camelyon17-id_test
              ]
            , val_dataset_length = 1024
            , test_dataset_length = 1024
            , num_adv = 128
            , w_perf = 0.2
            , w_rob = 0.2
            , w_sec = 0.2
            , w_cal = 0.2
            , w_oodd = 0.2
            }
          }

let HREConfig =
          SavingConfig::{=}
      /\  TrainingDatasetConfig::{=}
      /\  ResourceConfig::{=}
      /\  TrainingParamConfig::{=}
      /\  HRESetup::{=}

in  HREConfig
