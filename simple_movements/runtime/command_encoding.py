def condition(command):
    # The original 440D encoder and all widths are retained. Only current input
    # is repeated, never future joystick values, phase, or reference poses.
    result=command.new_zeros((len(command),10,44));result[:,:,:3]=command[:,None,:]
    return result.flatten(1)
