let HREConfig = ./default_config.dhall
in  [ HREConfig::{ algorithm = "erm" }
    , HREConfig::{=}
    , HREConfig::{ n_classes = 3 } ]
