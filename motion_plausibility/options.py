def add_bullet_args(parser):
    parser.add_argument(
        "--char_file", type=str, default="data/character/amass.urdf"
    )
    parser.add_argument("--char_info", type=str, default="amass_char_info.py")
    parser.add_argument("--scale", type=float, default=1.0)
    return parser


def add_preprocess_z_args(parser):
    parser.add_argument(
        "--z-input-path",
        type=str,
        required=True
    )
    parser.add_argument(
        "--stride", type=int, default=1,
    )
    return parser
