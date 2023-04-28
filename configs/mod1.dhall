let AdversarialAttack = (./enums.dhall).AdversarialAttack

let PGD = AdversarialAttack.PGD

let FGSM = AdversarialAttack.FGSM

let AutoAttack = AdversarialAttack.AutoAttack

let HREConfig = ./default_config.dhall
let HREConfig_t = (./config_types.dhall).HREConfig_t

let Union_t = < Val : Double | Expr : Text >

let HREConfigWAdvTraining =
          HREConfig
      //  { adversarial_training_method = Some PGD
          , adversarial_training_eps = Some (Union_t.Expr "3/255")
          }

let genAlgoName =
      \(config : HREConfig_t) ->
        let algo_name =
                  "AdvTraining_"

        in  config // { algorithm = algo_name }

let configs =
      [ genAlgoName (HREConfigWAdvTraining )
      , genAlgoName (HREConfigWAdvTraining // { adversarial_training_eps = Some (Union_t.Expr "1/255") } )
      , genAlgoName (HREConfigWAdvTraining // { adversarial_training_eps = Some (Union_t.Expr "7/255") } )
      , genAlgoName (HREConfigWAdvTraining // { adversarial_training_method = Some FGSM } )
      , genAlgoName (HREConfigWAdvTraining // { adversarial_training_method = Some AutoAttack } )
      ]


in  configs
