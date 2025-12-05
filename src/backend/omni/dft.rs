use crate::core::error::Result;
use crate::traits::dft::{Dft, DftPlan, DftSpec};
use std::marker::PhantomData;

pub struct OmniDft;

impl OmniDft {
    pub fn new() -> Self {
        Self
    }
}

struct OmniDftPlan<T> {
    spec: DftSpec,
    _marker: PhantomData<T>,
}

impl<T> DftPlan<T> for OmniDftPlan<T>
where T: Copy + Send + Sync 
{
    fn process(&self, input: &[T], output: &mut [T]) -> Result<()> {
        // Dummy implementation: just copy
        if input.len() != output.len() {
             return Err(crate::core::error::OmniError::BufferSizeMismatch { 
                 expected: input.len(), 
                 actual: output.len() 
             });
        }
        output.copy_from_slice(input);
        Ok(())
    }

    fn get_spec(&self) -> DftSpec {
        self.spec
    }
}

impl<T> Dft<T> for OmniDft 
where T: Copy + Send + Sync + 'static
{
    fn create_plan(&self, spec: DftSpec) -> Result<Box<dyn DftPlan<T>>> {
        Ok(Box::new(OmniDftPlan {
            spec,
            _marker: PhantomData,
        }))
    }
}
