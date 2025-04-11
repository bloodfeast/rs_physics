/// support point (for use with GJK)

#[derive(Debug, Clone)]
pub struct SupportPoint {
    pub point: (f64, f64, f64),
    pub point_a: (f64, f64, f64),
    pub point_b: (f64, f64, f64),
}

/// simplex (for use with GJK)
#[derive(Debug, Clone)]
pub struct Simplex {
    pub points: Vec<SupportPoint>,
}

impl Simplex {
    pub fn new() -> Self {
        Self { points: Vec::with_capacity(4) }
    }

    pub fn add(&mut self, point: SupportPoint) {
        self.points.push(point);
    }

    pub fn size(&self) -> usize {
        self.points.len()
    }

    pub fn get_a(&self) -> &SupportPoint {
        &self.points[self.points.len() - 1]
    }

    pub fn get_b(&self) -> &SupportPoint {
        &self.points[self.points.len() - 2]
    }

    pub fn get_c(&self) -> &SupportPoint {
        &self.points[self.points.len() - 3]
    }

    pub fn get_d(&self) -> &SupportPoint {
        &self.points[self.points.len() - 4]
    }

    pub fn set_a(&mut self, point: SupportPoint) {
        let last_idx = self.points.len() - 1;
        self.points[last_idx] = point;
    }

    pub fn set_abc(&mut self, a: SupportPoint, b: SupportPoint, c: SupportPoint) {
        self.points.clear();
        self.points.push(c);
        self.points.push(b);
        self.points.push(a);
    }

    pub fn set_ab(&mut self, a: SupportPoint, b: SupportPoint) {
        self.points.clear();
        self.points.push(b);
        self.points.push(a);
    }
}