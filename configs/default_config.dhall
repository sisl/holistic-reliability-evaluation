let Types = ./types.dhall

let user
    : Text
    = env:USER as Text

let pathMod
    : Text
    = "holistic-reliability-evaluation"

let scratchPath
    : Text
    = "/scratch/users/${user}/${pathMod}"

let SavingConfig_t =
      { Type = { algorithm : Text, phase : Types.Phase, save_folder : Text } }

let SavingConfig =
          SavingConfig_t
      /\  { default =
            { algorithm = "ERM"
            , phase = Types.Phase.train
            , save_folder = "\$scratchPath/results/"
            }
          }

let TrainingDatasetConfig_t =
      { Type =
          { data_dir : Text
          , n_classes : Natural
          , train_dataset : Types.Dataset
          , size : List Natural
          , n_channels : Natural
          , train_transforms : List Types.Transform
          , eval_transforms : List Types.Transform
          , min_performance : Double
          , max_performance : Double
          }
      }

let TrainingDatasetConfig =
          TrainingDatasetConfig_t
      /\  { default =
            { data_dir = "\$scratchPath/data/"
            , n_classes = 182
            , train_dataset = Types.Dataset.iwildcam-train
            , size = [ 224, 224 ]
            , n_channels = 3
            , train_transforms = [] : List Types.Transform
            , eval_transforms = [ Types.Transform.wilds_default_normalization ]
            , min_performance = 0.0
            , max_performance = 1.0
            }
          }

let ResourceConfig_t =
      { Type =
          { max_num_workers : Natural
          , accelerator : Types.Accelerator
          , devices : Natural
          }
      }

let ResourceConfig =
          ResourceConfig_t
      /\  { default =
            { max_num_workers = 32
            , accelerator = Types.Accelerator.gpu
            , devices = 1
            }
          }

let TrainingParamConfig_t =
      { Type =
          { max_epochs : Natural
          , lr : Double
          , batch_size : Natural
          , optimizer : Types.Optim
          , model_source : Types.ModelLibs
          , model : Types.Models
          , label_smoothing : Double
          , pretrained_weights : Types.PretrainedWeights
          , freeze_weights : Bool
          , unfreeze_k_layers : Natural
          , calibration_method : Optional Types.CalibrationMethod
          , adversarial_training_method :
              Optional Types.AdversarialAttackMethods
          , adversarial_training_eps : Optional < Double | Text >
          }
      }

let TrainingParamConfig =
          TrainingParamConfig_t
      /\  { default =
            { max_epochs = 12
            , lr = 0.001
            , batch_size = 24
            , optimizer = Types.Optim.adam
            , model_source = Types.ModelLibs.torchvision
            , model = Types.Models.resnet50
            , label_smoothing = 0.0
            , pretrained_weights = Types.PretrainedWeights.DEFAULT
            , freeze_weights = False
            , unfreeze_k_layers = 0
            , calibration_method = None Types.CalibrationMethod
            , adversarial_training_method = None Types.AdversarialAttackMethods
            , adversarial_training_eps = None < Double | Text >
            }
          }

let HRESetup_t =
      { Type =
          { val_id_dataset : List Types.Dataset
          , val_ds_datasets : List Types.Dataset
          , val_ood_datasets : List Types.Dataset
          , test_id_dataset : List Types.Dataset
          , test_ds_datasets : List Types.Dataset
          , test_ood_datasets : List Types.Dataset
          , val_dataset_length : Natural
          , test_dataset_length : Natural
          , num_adv : Natural
          , w_perf : Double
          , w_rob : Double
          , w_sec : Double
          , w_cal : Double
          , w_oodd : Double
          }
      }

let HRESetup =
          HRESetup_t
      /\  { default =
            { val_id_dataset = [ Types.Dataset.iwildcam-id_val ]
            , val_ds_datasets =
              [ Types.Dataset.iwildcam-val
              , Types.Dataset.iwildcam-id_val-corruption1_val
              ]
            , val_ood_datasets =
              [ Types.Dataset.gaussian_noise
              , Types.Dataset.fmow-id_val
              , Types.Dataset.rxrx1-id_val
              , Types.Dataset.camelyon17-id_val
              ]
            , test_id_dataset = [ Types.Dataset.iwildcam-id_test ]
            , test_ds_datasets =
              [ Types.Dataset.iwildcam-test
              , Types.Dataset.iwildcam-id_test-corruption1_test
              ]
            , test_ood_datasets =
              [ Types.Dataset.gaussian_noise
              , Types.Dataset.fmow-id_test
              , Types.Dataset.rxrx1-id_test
              , Types.Dataset.camelyon17-id_test
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
