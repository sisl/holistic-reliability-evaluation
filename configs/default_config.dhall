let Types = ./types.dhall

let HREConfig =
      { Type =
          { algorithm : Text
          , phase : Types.Phase
          , save_folder : Text
          , data_dir : Text
          , n_classes : Natural
          , train_dataset : Types.Dataset
          , size : List Natural
          , train_transforms : List Types.Transform
          , n_channels : Natural
          , min_performance : Double
          , max_performance : Double
          , max_num_workers : Natural
          , accelerator : Types.Accellerator
          , devices : Natural
          }
      , default =
          let user
              : Text
              = env:USER as Text
          let pathMod : Text = "holistic-reliability-evaluation"

          in  { algorithm = "ERM"
              , phase = Types.Phase.train
              , save_folder = "/scratch/users/${user}/${pathMod}/results/"
              , data_dir = "/scratch/users/${user}/${pathMod}/data/"
              , n_classes = 3
              , n_channels = 3
              , train_dataset = Types.Dataset.iwildcam-train
              , size = [ 224, 224 ]
              , min_performance = 0.0
              , max_performance = 1.0
              , max_num_workers = 8
              , accelerator = Types.Accellerator.gpu
              , train_transforms = [ Types.Transform.default ]
              , devices = 1
              }
      }
in HREConfig
