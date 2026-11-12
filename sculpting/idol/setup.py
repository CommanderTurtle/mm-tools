from state import ROOT, SOURCE, MODELS, require_models


def main():
    require_models()
    if not (SOURCE / "idol" / "__init__.py").is_file() or not (SOURCE / "pyproject.toml").is_file():
        raise SystemExit("Missing committed idol-sdk source under " + str(SOURCE))
    link = ROOT / "models"
    if link.is_symlink() and link.resolve() == MODELS.resolve():
        return
    if link.exists() or link.is_symlink():
        raise SystemExit(
            "Preserving existing models path; expected a link to " + str(MODELS)
        )
    link.symlink_to(os.path.relpath(MODELS, ROOT), target_is_directory=True)


if __name__ == "__main__":
    main()
