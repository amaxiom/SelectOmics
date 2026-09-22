"""
SelectOmics command-line interface.

Usage examples
--------------
    # Suggest a config based on your data:
    selectomics suggest data.csv --target Disease

    # Generate a commented YAML template:
    selectomics template
    selectomics template --output my_config.yaml

    # Run the pipeline with a preset:
    selectomics run data.csv --target Disease --preset standard

    # Run the pipeline with a YAML config:
    selectomics run --config my_config.yaml

    # Run and save results:
    selectomics run data.csv --target Disease --preset quick --output results/
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _setup_logging(verbose: bool = False) -> None:
    """Configure console logging for the CLI."""
    import SelectOmics
    level = 'DEBUG' if verbose else 'INFO'
    SelectOmics.enable_logging(level)


def cmd_suggest(args: argparse.Namespace) -> int:
    """Inspect a dataset and print a recommended configuration."""
    _setup_logging(args.verbose)
    from SelectOmics.config import SelectOmicsConfig

    overrides = _parse_overrides(args.set)

    cfg = SelectOmicsConfig.suggest(
        data_path=args.data,
        target_column=args.target,
        **overrides,
    )

    if args.output:
        cfg.to_yaml(args.output)
        print(f"\nSuggested config written to: {args.output}")
        print("Review and edit it, then run:")
        print(f"  selectomics run --config {args.output}")
    else:
        print("\nSuggested config (add --output config.yaml to save):")
        # Print key fields
        d = cfg.to_dict()
        important = [
            'algorithm', 'n_consensus_models', 'quick_tune_iterations',
            'enable_step3', 'n_bootstrap', 'enable_nested_cv',
            'enable_holdout_ranking', 'holdout_ranking_splits',
        ]
        for k in important:
            print(f"  {k}: {d[k]}")

    return 0


def cmd_template(args: argparse.Namespace) -> int:
    """Generate a commented YAML configuration template."""
    from SelectOmics.config import SelectOmicsConfig

    template = SelectOmicsConfig.template_yaml(path=args.output)

    if args.output:
        print(f"Template written to: {args.output}")
        print(f"Edit it, then run: selectomics run --config {args.output}")
    else:
        print(template)

    return 0


def cmd_run(args: argparse.Namespace) -> int:
    """Execute the SelectOmics pipeline."""
    _setup_logging(args.verbose)

    import SelectOmics
    from SelectOmics.config import SelectOmicsConfig

    overrides = _parse_overrides(args.set)

    # Build config from --config file, --preset, or positional data arg.
    if args.config:
        cfg_path = Path(args.config)
        try:
            if cfg_path.suffix.lower() in ('.yaml', '.yml'):
                cfg = SelectOmicsConfig.from_yaml(cfg_path)
            else:
                cfg = SelectOmicsConfig.from_json(cfg_path)
            if overrides:
                d = cfg.to_dict()
                d.update(overrides)
                cfg = SelectOmicsConfig.from_dict(d)
        except (FileNotFoundError, ValueError) as exc:
            # Reported the way the preset and argument errors below are. A
            # missing or malformed config file is a user error, not a defect,
            # and a traceback tells the user nothing they can act on.
            print(f"ERROR: Could not load config '{cfg_path}': {exc}",
                  file=sys.stderr)
            if args.verbose:
                raise
            return 1
    elif args.data and args.target:
        preset = args.preset or 'standard'
        preset_fn = {
            'quick':    SelectOmicsConfig.quick,
            'standard': SelectOmicsConfig.standard,
            'thorough': SelectOmicsConfig.thorough,
            'omics':    SelectOmicsConfig.omics,
        }.get(preset)
        if preset_fn is None:
            print(f"ERROR: Unknown preset '{preset}'. Choose: quick, standard, thorough, omics",
                  file=sys.stderr)
            return 1
        cfg = preset_fn(args.data, args.target, **overrides)
    else:
        print(
            "ERROR: Provide either --config <file.yaml> or <data.csv> --target <column>",
            file=sys.stderr,
        )
        return 1

    # Override output dir if provided.
    if args.output:
        d = cfg.to_dict()
        d['output_dir'] = args.output
        cfg = SelectOmicsConfig.from_dict(d)

    # Run.
    #
    # The branches above report their failures cleanly and exit 1; this one
    # did not, so everything the pipeline raises arrived as a traceback. The
    # common cases are all ordinary user errors: a data file that is not
    # there, a target column that is not in it, a target whose classes have
    # one member each. The exit code is unchanged, since an unhandled
    # exception exits 1 already; --verbose re-raises for the full traceback.
    pipeline = SelectOmics.SelectOmicsPipeline(cfg)
    try:
        results = pipeline.run(resume=args.resume)
        pipeline.save_results()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if args.verbose:
            raise
        return 1

    # Print summary.
    selected = pipeline.get_selected_features()
    print(f"\n{'='*50}")
    print("SelectOmics complete.")
    print(f"  Selected features: {len(selected)} (last step)")
    # The last step and the recommended panel differ when a later step pruned
    # past the point where it helped, and a summary naming only the first
    # sent CLI users to the panel validation had advised against.
    rec = results.get('recommendation')
    if rec is not None:
        print(f"  Recommended:       {rec['n_features']} features, "
              f"{rec['step_name']} (recommended_features file)")
    print(f"  Results saved to:  {cfg.output_dir}")
    if 'final_test' in results:
        ft = results['final_test']
        auc = ft.get('test_auc', float('nan'))
        print(f"  Final test AUC:    {auc:.4f} "
              f"({ft.get('panel', 'last step')} panel)")
    print(f"{'='*50}")

    return 0


def _parse_overrides(set_args: list | None) -> dict:
    """
    Parse --set key=value pairs into a dict.

    Values are coerced: 'true'/'false' -> bool, 'none'/'null'/empty -> None,
    numeric strings -> int/float, everything else stays as a string.

    The None coercion matters for the Optional fields ``min_consensus`` and
    ``stability_threshold``, where None is a meaningful setting rather than an
    absence: for min_consensus it means "relax as far as min_features_floor
    requires". Without it, ``--set min_consensus=none`` passed the string
    'none' into validation and raised an unhandled TypeError from comparing a
    float against a str, rather than either working or failing cleanly.
    """
    overrides: dict = {}
    if not set_args:
        return overrides
    for item in set_args:
        if '=' not in item:
            print(f"WARNING: Ignoring malformed --set argument: {item!r} (expected key=value)",
                  file=sys.stderr)
            continue
        key, _, value = item.partition('=')
        key = key.strip()
        value = value.strip()
        # Type coercion.
        if value.lower() == 'true':
            overrides[key] = True
        elif value.lower() == 'false':
            overrides[key] = False
        elif value.lower() in ('none', 'null', ''):
            overrides[key] = None
        else:
            try:
                overrides[key] = int(value)
            except ValueError:
                try:
                    overrides[key] = float(value)
                except ValueError:
                    overrides[key] = value
    return overrides


def build_parser() -> argparse.ArgumentParser:
    """
    Construct the selectomics argument parser.

    Separated from ``main`` so tests can inspect the parsed namespace without
    executing a command.
    """
    try:
        import SelectOmics
        _version = SelectOmics.__version__
    except Exception:
        _version = 'unknown'
    parser = argparse.ArgumentParser(
        prog='selectomics',
        description='SelectOmics: feature selection pipeline for high-dimensional omics data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  selectomics suggest data.csv --target Disease
  selectomics suggest data.csv --target Disease --output config.yaml
  selectomics template --output config.yaml
  selectomics run data.csv --target Disease --preset standard
  selectomics run data.csv --target Disease --preset quick --set algorithm=RF
  selectomics run --config config.yaml
  selectomics run --config config.yaml --resume
        """,
    )
    # --verbose is accepted both before and after the subcommand.  Declaring it
    # on the subparsers with a store_true default would reset the flag set on
    # the main parser, so the subcommand copies default to SUPPRESS and the
    # main parser holds the only default.
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable DEBUG-level logging.')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {_version}')

    subparsers = parser.add_subparsers(dest='command', required=True)

    # --- suggest ---
    p_suggest = subparsers.add_parser(
        'suggest',
        help='Inspect a dataset and recommend a starting configuration.',
    )
    p_suggest.add_argument('data', help='Path to input data file.')
    p_suggest.add_argument('--target', '-t', required=True,
                           help='Target column name.')
    p_suggest.add_argument('--output', '-o',
                           help='Write suggested config to this YAML file.')
    p_suggest.add_argument('--set', metavar='KEY=VALUE', action='append',
                           help='Override suggested field (e.g. --set algorithm=RF). Repeatable.')
    p_suggest.add_argument('--verbose', '-v', action='store_true',
                           default=argparse.SUPPRESS)
    p_suggest.set_defaults(func=cmd_suggest)

    # --- template ---
    p_template = subparsers.add_parser(
        'template',
        help='Generate a fully-commented YAML configuration template.',
    )
    p_template.add_argument('--output', '-o',
                            help='Write template to this file (default: print to stdout).')
    p_template.set_defaults(func=cmd_template)

    # --- run ---
    p_run = subparsers.add_parser(
        'run',
        help='Execute the SelectOmics pipeline.',
    )
    p_run.add_argument('data', nargs='?',
                       help='Path to input data file (required unless --config is given).')
    p_run.add_argument('--target', '-t',
                       help='Target column name (required unless --config is given).')
    p_run.add_argument('--preset', '-p',
                       choices=['quick', 'standard', 'thorough', 'omics'],
                       default='standard',
                       help='Starting configuration preset (default: standard). '
                            "'omics' uses the benchmark-tuned small-n/large-p thresholds.")
    p_run.add_argument('--config', '-c',
                       help='Path to a YAML or JSON config file (overrides --preset).')
    p_run.add_argument('--output', '-o',
                       help='Override output_dir in the config.')
    p_run.add_argument('--resume', action='store_true',
                       help='Resume from an existing checkpoint.')
    p_run.add_argument('--set', metavar='KEY=VALUE', action='append',
                       help='Override any config field (e.g. --set n_consensus_models=3). Repeatable.')
    p_run.add_argument('--verbose', '-v', action='store_true',
                       default=argparse.SUPPRESS)
    p_run.set_defaults(func=cmd_run)

    return parser


def main(argv: list | None = None) -> int:
    """Entry point for the selectomics CLI."""
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
