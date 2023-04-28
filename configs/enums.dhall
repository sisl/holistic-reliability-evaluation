{ Phase = < train | validation >
, Dataset =
    < iwildcam-train
    | iwildcam-id_val
    | iwildcam-val
    | iwildcam-id_val-corruption1_val
    | gaussian_noise
    | fmow-id_val
    | rxrx1-id_val
    | fmow-id_test
    | rxrx1-id_test
    | camelyon17-id_test
    | camelyon17-id_val
    | iwildcam-id_test
    | iwildcam-test
    | iwildcam-id_test-corruption1_test
    >
, Accelerator = < cpu | gpu >
, Transform = < default | wilds_default_normalization >
, Optim = < adam | adamw | sgd >
, ModelLibs = < torchvision | open_clip >
, Models = < resnet50 >
, PretrainedWeights = < DEFAULT >
, AdversarialAttack = < PGD | FGSM | AutoAttack >
, CalibrationMethod = < temperature_scaling >
}
