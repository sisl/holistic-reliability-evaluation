let Enums = ./enums.dhall

let Phase = Enums.Phase

let Dataset = Enums.Dataset

let Accelerator = Enums.Accelerator

let Transform = Enums.Transform

let Optim = Enums.Optim

let ModelLibs = Enums.ModelLibs

let Models = Enums.Models

let PretrainedWeights = Enums.PretrainedWeights

let AdversarialAttack = Enums.AdversarialAttack

let CalibrationMethod = Enums.CalibrationMethod

let SavingConfig_t =
      { Type = { algorithm : Text, phase : Phase, save_folder : Text } }

let TrainingDatasetConfig_t =
      { Type =
          { data_dir : Text
          , n_classes : Natural
          , train_dataset : Dataset
          , size : List Natural
          , n_channels : Natural
          , train_transforms : List Transform
          , eval_transforms : List Transform
          , min_performance : Double
          , max_performance : Double
          }
      }

let ResourceConfig_t =
      { Type =
          { max_num_workers : Natural
          , accelerator : Accelerator
          , devices : Natural
          }
      }

let Union_t = < Val : Double | Expr : Text >

let TrainingParamConfig_t =
      { Type =
          { max_epochs : Natural
          , lr : Double
          , batch_size : Natural
          , optimizer : Optim
          , model_source : ModelLibs
          , model : Models
          , label_smoothing : Double
          , pretrained_weights : PretrainedWeights
          , freeze_weights : Bool
          , unfreeze_k_layers : Natural
          , calibration_method : Optional CalibrationMethod
          , adversarial_training_method : Optional AdversarialAttack
          , adversarial_training_eps : Optional Union_t
          }
      }

let HRESetup_t =
      { Type =
          { val_id_dataset : Dataset
          , val_ds_datasets : List Dataset
          , val_ood_datasets : List Dataset
          , test_id_dataset : Dataset
          , test_ds_datasets : List Dataset
          , test_ood_datasets : List Dataset
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

in  { SavingConfig_t
    , TrainingDatasetConfig_t
    , ResourceConfig_t
    , TrainingParamConfig_t
    , HRESetup_t
    , HREConfig_t =
            SavingConfig_t.Type
        //\\  TrainingDatasetConfig_t.Type
        //\\  ResourceConfig_t.Type
        //\\  TrainingParamConfig_t.Type
        //\\  HRESetup_t.Type
    }
