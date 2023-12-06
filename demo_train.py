
from gluefactory.train import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, default="sp+lg_megadepth")
    parser.add_argument("--conf", type=str, default="gluefactory/configs/superpoint+lightglue_megadepth.yaml")
    parser.add_argument(
        "--mixed_precision",
        "--mp",
        default=None,
        type=str,
        choices=["float16", "bfloat16"],
    )
    parser.add_argument(
        "--compile",
        default=None,
        type=str,
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--restore", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--print_arch", "--pa", action="store_true")
    parser.add_argument("--detect_anomaly", "--da", action="store_true")
    parser.add_argument("--log_it", "--log_it", action="store_true")
    parser.add_argument("--no_eval_0", action="store_true")
    parser.add_argument("--run_benchmarks", action="store_true")
    # parser.add_argument("dotlist", nargs="*")
    parser.add_argument("--dotlist", type=list, default=[])
    args = parser.parse_intermixed_args()

    logger.info(f"Starting experiment: {args.experiment}")
    output_dir = Path(TRAINING_PATH, args.experiment)
    output_dir.mkdir(exist_ok=True, parents=True)

    print('args.dotlist: ', args.dotlist)
    conf = OmegaConf.from_cli(args.dotlist)
    if args.conf:
        conf = OmegaConf.merge(OmegaConf.load(args.conf), conf)
    elif args.restore:
        restore_conf = OmegaConf.load(output_dir / "config.yaml")
        conf = OmegaConf.merge(restore_conf, conf)
    if not args.restore:
        if conf.train.seed is None:
            conf.train.seed = torch.initial_seed() & (2**32 - 1)
        OmegaConf.save(conf, str(output_dir / "config.yaml"))

    # # copy gluefactory and submodule into output dir
    # for module in conf.train.get("submodules", []) + [__module_name__]:
    #     mod_dir = Path(__import__(str(module)).__file__).parent
    #     shutil.copytree(mod_dir, output_dir / module, dirs_exist_ok=True)

    if args.distributed:
        args.n_gpus = torch.cuda.device_count()
        args.lock_file = output_dir / "distributed_lock"
        if args.lock_file.exists():
            args.lock_file.unlink()
        torch.multiprocessing.spawn(
            main_worker, nprocs=args.n_gpus, args=(conf, output_dir, args)
        )
    else:
        main_worker(0, conf, output_dir, args)