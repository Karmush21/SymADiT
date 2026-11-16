from src.models.encoders.sym_transformer_encoder import SymTransformerEncoder
from src.models.decoders.sym_transformer_decoder import SymTransformerDecoder
from src.models.ae_module import SymAutoencoder

def create_random_autoencoder():
    '''
        Purely for debugging
    '''
    encoder = SymTransformerEncoder(
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        num_layers=6,
    )

    decoder = SymTransformerDecoder(
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        num_layers=6,
    )

    autoencoder = SymAutoencoder(
        encoder=encoder,
        decoder=decoder,

        pos_loss="l2",
        cell_loss="l2",

        latent_dim=32,

        optimizer=None,
        scheduler=None,
        scheduler_frequency=1,

        use_regularized_ae=False,
        use_regularized_kl=False,

        loss_reg_terms={
            "duplicate_zero_dof_wyckoff": False,
            "duplicate_same_wp_reps": False,
            "num_atoms": False,
            "expand_crys_reg": {
                "do_expand_crys_reg": False,
                "act_func": "relu",
            },
        },

        loss_weights={
            "loss_atom_types": 1.0,
            "loss_wyckoff": 1.0,
            "loss_pos": 1.0,
            "loss_cell": 1.0,
            "loss_kl": 0.0,
            "loss_num_atoms": 0.0,
            "loss_too_close": 0.0,
            "loss_duplicate_same_wp_reps": 0.0,
        },

        augmentations={
            "do_augmentations": False,
            "pos_sigma": 0.0,
            "cell_len_sigma": 0.0,
            "cell_ang_sigma": 0.0,
        },

        compile=False,  
        debug=True,
    )

    return autoencoder