use version::GroveVersion;
use versioned_feature_core::FeatureVersion;

pub mod error;
pub mod version;

/// The `UnknownVersionMismatch` error every version-check macro below
/// returns: built out of line so the checks themselves stay a comparison and
/// a return.
#[cold]
#[doc(hidden)]
pub fn unknown_version_mismatch(
    method: &str,
    known_versions: &[FeatureVersion],
    received: FeatureVersion,
) -> error::GroveVersionError {
    error::GroveVersionError::UnknownVersionMismatch {
        method: method.to_string(),
        known_versions: known_versions.to_vec(),
        received,
    }
}

#[macro_export]
macro_rules! check_grovedb_v0_with_cost {
    ($method:expr, $version:expr) => {{
        const EXPECTED_VERSION: u16 = 0;
        if $version != EXPECTED_VERSION {
            return grovedb_costs::CostsExt::wrap_with_cost(
                Err(
                    $crate::unknown_version_mismatch($method, &[EXPECTED_VERSION], $version).into(),
                ),
                Default::default(),
            );
        }
    }};
}

#[macro_export]
macro_rules! check_grovedb_v1_with_cost {
    ($method:expr, $version:expr) => {{
        const EXPECTED_VERSION: u16 = 1;
        if $version != EXPECTED_VERSION {
            return grovedb_costs::CostsExt::wrap_with_cost(
                Err(
                    $crate::unknown_version_mismatch($method, &[EXPECTED_VERSION], $version).into(),
                ),
                Default::default(),
            );
        }
    }};
}

#[macro_export]
macro_rules! check_grovedb_v0 {
    ($method:expr, $version:expr) => {{
        const EXPECTED_VERSION: u16 = 0;
        if $version != EXPECTED_VERSION {
            return Err(
                $crate::unknown_version_mismatch($method, &[EXPECTED_VERSION], $version).into(),
            );
        }
    }};
}

#[macro_export]
macro_rules! check_grovedb_v0_or_v1_with_cost {
    ($method:expr, $version:expr) => {{
        const EXPECTED_VERSION_V0: u16 = 0;
        const EXPECTED_VERSION_V1: u16 = 1;
        if $version != EXPECTED_VERSION_V0 && $version != EXPECTED_VERSION_V1 {
            return grovedb_costs::CostsExt::wrap_with_cost(
                Err($crate::unknown_version_mismatch(
                    $method,
                    &[EXPECTED_VERSION_V0, EXPECTED_VERSION_V1],
                    $version,
                )
                .into()),
                Default::default(),
            );
        }
    }};
}

#[macro_export]
macro_rules! check_grovedb_v0_or_v1 {
    ($method:expr, $version:expr) => {{
        const EXPECTED_VERSION_V0: u16 = 0;
        const EXPECTED_VERSION_V1: u16 = 1;
        if $version != EXPECTED_VERSION_V0 && $version != EXPECTED_VERSION_V1 {
            return Err($crate::unknown_version_mismatch(
                $method,
                &[EXPECTED_VERSION_V0, EXPECTED_VERSION_V1],
                $version,
            )
            .into());
        }
        $version
    }};
}

#[macro_export]
macro_rules! check_grovedb_v0_v1_or_v2 {
    ($method:expr, $version:expr) => {{
        const EXPECTED_VERSION_V0: u16 = 0;
        const EXPECTED_VERSION_V1: u16 = 1;
        const EXPECTED_VERSION_V2: u16 = 2;
        if $version != EXPECTED_VERSION_V0
            && $version != EXPECTED_VERSION_V1
            && $version != EXPECTED_VERSION_V2
        {
            return Err($crate::unknown_version_mismatch(
                $method,
                &[
                    EXPECTED_VERSION_V0,
                    EXPECTED_VERSION_V1,
                    EXPECTED_VERSION_V2,
                ],
                $version,
            )
            .into());
        }
        $version
    }};
}

/// Dispatches a cost-returning method body by feature version. Unknown
/// versions return `UnknownVersionMismatch` wrapped in a `CostResult`, so
/// this belongs in functions returning `CostResult<_, _>`.
#[macro_export]
macro_rules! dispatch_version {
    ($method:expr, $version:expr, $($($version_num:literal)|+ => {$($body:tt)*})+) => {
        {
            let version = $version;
            if $($(version != $version_num)&&*)&&* {
                return grovedb_costs::CostsExt::wrap_with_cost(
                    Err($crate::unknown_version_mismatch($method, &[$($($version_num),*),*], $version)
                    .into()),
                    Default::default(),
                );
            }

            match version {
                $($($version_num)|+ => {$($body)*})*
                _ => unreachable!()
            }
        }
    };
}

#[macro_export]
macro_rules! check_merk_v0_with_cost {
    ($method:expr, $version:expr) => {{
        const EXPECTED_VERSION: u16 = 0;
        if $version != EXPECTED_VERSION {
            return grovedb_costs::CostsExt::wrap_with_cost(
                Err(
                    $crate::unknown_version_mismatch($method, &[EXPECTED_VERSION], $version).into(),
                ),
                Default::default(),
            );
        }
    }};
}

#[macro_export]
macro_rules! check_merk_v0 {
    ($method:expr, $version:expr) => {{
        const EXPECTED_VERSION: u16 = 0;
        if $version != EXPECTED_VERSION {
            return Err(
                $crate::unknown_version_mismatch($method, &[EXPECTED_VERSION], $version).into(),
            );
        }
    }};
}

pub trait TryFromVersioned<T>: Sized {
    /// The type returned in the event of a conversion error.
    type Error;

    /// Performs the conversion.
    fn try_from_versioned(value: T, grove_version: &GroveVersion) -> Result<Self, Self::Error>;
}

pub trait TryIntoVersioned<T>: Sized {
    /// The type returned in the event of a conversion error.
    type Error;

    /// Performs the conversion.
    fn try_into_versioned(self, grove_version: &GroveVersion) -> Result<T, Self::Error>;
}

impl<T, U> TryIntoVersioned<U> for T
where
    U: TryFromVersioned<T>,
{
    type Error = U::Error;

    #[inline]
    fn try_into_versioned(self, grove_version: &GroveVersion) -> Result<U, U::Error> {
        U::try_from_versioned(self, grove_version)
    }
}

impl<T, U> TryFromVersioned<U> for T
where
    T: TryFrom<U>,
{
    type Error = T::Error;

    #[inline]
    fn try_from_versioned(value: U, _grove_version: &GroveVersion) -> Result<Self, Self::Error> {
        T::try_from(value)
    }
}

#[cfg(test)]
mod tests;
