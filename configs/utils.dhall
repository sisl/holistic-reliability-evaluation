let user
    : Text
    = env:USER as Text

let pathMod
    : Text
    = "holistic-reliability-evaluation"

let scratchPath
    : Text
    = "/scratch/users/${user}/${pathMod}"
in {scratchPath}
