import React, { useState } from 'react';
import axios from 'axios';
import { Table, Button, Alert, Card, Container, Row, Col, Spinner, Badge, Form, Tab, Tabs } from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
 
function App() {
  // State for file upload
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('');
  const [data, setData] = useState(null);
 
  // State for customer ID
  const [customerId, setCustomerId] = useState('');
 
  // Common states for both modes
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState('file');
 
  const API_URL = 'http://localhost:8000/predict';
 
  const badgeColors = {
    Platinum: "primary",  // Blue
    Gold: "warning",      // Yellow
    Silver: "secondary",  // Grey
    Bronze: "danger",     // Red
    Copper: "dark"        // Dark Grey
  };
 
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile && selectedFile.name.endsWith('.csv')) {
      setFile(selectedFile);
      setFileName(selectedFile.name);
      setError('');
     
      // Read and display the CSV
      const reader = new FileReader();
      reader.onload = (event) => {
        const csvData = event.target.result;
        const parsedData = parseCSV(csvData);
        setData(parsedData);
      };
      reader.readAsText(selectedFile);
    } else {
      setFile(null);
      setFileName('');
      setData(null);
      setError('Please upload a CSV file');
    }
  };
 
  const parseCSV = (text) => {
    const lines = text.split('\n');
    const headers = lines[0].split(',');
   
    const parsedData = [];
    for (let i = 1; i < lines.length && i < 10; i++) {
      if (lines[i].trim() === '') continue;
     
      const values = lines[i].split(',');
      const row = {};
     
      headers.forEach((header, index) => {
        row[header.trim()] = values[index]?.trim() || '';
      });
     
      parsedData.push(row);
    }
   
    return { headers, rows: parsedData };
  };
 
  const handleCustomerIdChange = (e) => {
    setCustomerId(e.target.value);
    setError('');
  };
 
  const handleFileSubmit = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }
 
    setLoading(true);
    setError('');
    setResult(null);
 
    const formData = new FormData();
    formData.append('file', file);
 
    try {
      const response = await axios.post(API_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        }
      });
     
      setResult(response.data);
    } catch (err) {
      console.error('Error:', err);
      setError(`Error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };
 
  const handleCustomerIdSubmit = async () => {
    if (!customerId.trim()) {
      setError('Please enter a Customer ID');
      return;
    }
 
    setLoading(true);
    setError('');
    setResult(null);
 
    try {
      const response = await axios.post(`${API_URL}?customerID=${customerId}`);
      setResult(response.data);
    } catch (err) {
      console.error('Error:', err);
      setError(`Error: ${err.response?.data?.detail || err.message}`);
    } finally {
      setLoading(false);
    }
  };
 
  const downloadCSV = () => {
    if (!result || !result.predictions) return;
   
    const headers = Object.keys(result.predictions[0]).join(',');
    const rows = result.predictions.map(row => Object.values(row).join(',')).join('\n');
    const csvContent = `${headers}\n${rows}`;
   
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = "Results.csv";
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };
 
  return (
    <Container className="mt-5">
      <Card className="shadow-sm">
        <Card.Header as="h2" className="text-center bg-primary text-white">
          Bajaj Bounce Back Prediction
        </Card.Header>
        <Card.Body>
          <Tabs
            activeKey={activeTab}
            onSelect={(k) => {
              setActiveTab(k);
              setError('');
              setResult(null);
            }}
            className="mb-4"
          >
            <Tab eventKey="file" title="Upload CSV">
              <Row className="mb-4 mt-3">
                <Col>
                  <div className="mb-3">
                    <Form.Group controlId="csvFile">
                      <Form.Label>Upload a CSV file</Form.Label>
                      <Form.Control
                        type="file"
                        accept=".csv"
                        onChange={handleFileChange}
                      />
                    </Form.Group>
                  </div>
                 
                  {fileName && (
                    <Alert variant="success">
                      File uploaded successfully: {fileName}
                    </Alert>
                  )}
                 
                  <Button
                    variant="primary"
                    onClick={handleFileSubmit}
                    disabled={!file || loading}
                    className="mb-3"
                  >
                    {loading ? (
                      <>
                        <Spinner as="span" animation="border" size="sm" />
                        Processing...
                      </>
                    ) : (
                      'Predict'
                    )}
                  </Button>
                </Col>
              </Row>
             
              {data && (
                <Row className="mb-4">
                  <Col>
                    <h4>Uploaded Data Preview:</h4>
                    <div className="table-responsive">
                      <Table striped bordered hover size="sm">
                        <thead>
                          <tr>
                            {data.headers.map((header, idx) => (
                              <th key={idx}>{header}</th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {data.rows.map((row, rowIdx) => (
                            <tr key={rowIdx}>
                              {data.headers.map((header, colIdx) => (
                                <td key={colIdx}>{row[header]}</td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    </div>
                    <small className="text-muted">Showing first 10 rows only</small>
                  </Col>
                </Row>
              )}
            </Tab>
           
            <Tab eventKey="customerId" title="Search by Customer ID">
              <Row className="mb-4 mt-3">
                <Col>
                  <Form.Group className="mb-3" controlId="customerIdInput">
                    <Form.Label>Enter Customer ID</Form.Label>
                    <Form.Control
                      type="text"
                      placeholder="e.g., CUST-49028"
                      value={customerId}
                      onChange={handleCustomerIdChange}
                    />
                  </Form.Group>
                 
                  <Button
                    variant="primary"
                    onClick={handleCustomerIdSubmit}
                    disabled={!customerId.trim() || loading}
                    className="mb-3"
                  >
                    {loading ? (
                      <>
                        <Spinner as="span" animation="border" size="sm" />
                        Processing...
                      </>
                    ) : (
                      'Search'
                    )}
                  </Button>
                </Col>
              </Row>
            </Tab>
          </Tabs>
         
          {error && <Alert variant="danger">{error}</Alert>}
         
          {result && activeTab === 'file' && result.predictions && (
            <>
              <Row className="mb-4">
                <Col>
                  <h4>Results Summary:</h4>
                  <Card className="text-center mb-3">
                    <Card.Body>
                      <Row>
                        <Col>
                          <h5>Platinum</h5>
                          <h2>
                            <Badge bg="primary">{result.summary.platinum_predictions}</Badge>
                          </h2>
                        </Col>
                        <Col>
                          <h5>Gold</h5>
                          <h2>
                            <Badge bg="warning">{result.summary.glod_predictions}</Badge>
                          </h2>
                        </Col>
                        <Col>
                          <h5>Silver</h5>
                          <h2>
                            <Badge bg="secondary">{result.summary.silver_predictions}</Badge>
                          </h2>
                        </Col>
                        <Col>
                          <h5>Bronze</h5>
                          <h2>
                            <Badge bg="danger">{result.summary.bronze_predictions}</Badge>
                          </h2>
                        </Col>
                        <Col>
                          <h5>Copper</h5>
                          <h2>
                            <Badge bg="dark">{result.summary.copper_predictions}</Badge>
                          </h2>
                        </Col>
                      </Row>
                      <Row className="mt-3">
                        <Col>
                          <h5>Total Predictions</h5>
                          <h2>
                            <Badge bg="info">{result.summary.total_predictions}</Badge>
                          </h2>
                        </Col>
                      </Row>
                    </Card.Body>
                  </Card>
                </Col>
              </Row>
             
              <Row>
                <Col>
                  <h4>Prediction Results:</h4>
                  <div className="table-responsive">
                    <Table striped bordered hover size="sm">
                      <thead>
                        <tr>
                          {Object.keys(result.predictions[0]).map((key, idx) => (
                            <th key={idx}>{key}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {result.predictions.slice(0, 10).map((row, rowIdx) => (
                          <tr key={rowIdx}>
                            {Object.keys(row).map((key, colIdx) => (
                              <td key={colIdx}>
                                {key === 'Label' ? (
                                  <Badge bg={badgeColors[row[key]] || "light"}>
                                    {row[key]}
                                  </Badge>
                                ) : (
                                  row[key]
                                )}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </Table>
                  </div>
                  <small className="text-muted">Showing first 10 rows only</small>
                </Col>
              </Row>
             
              <Row className="mt-3">
                <Col className="text-center">
                  <Button variant="success" onClick={downloadCSV}>
                    Download Full Results
                  </Button>
                </Col>
              </Row>
            </>
          )}
         
          {result && activeTab === 'customerId' && result.customer_prediction && (
            <Row className="mb-4">
              <Col>
                <h4>Customer Details:</h4>
                <Card>
                  <Card.Body>
                    <Row className="mb-3">
                      <Col sm={3}><strong>Customer ID:</strong></Col>
                      <Col>{result.customer_prediction.customer_ID}</Col>
                    </Row>
                    <Row className="mb-3">
                      <Col sm={3}><strong>Score:</strong></Col>
                      <Col>
                        <Badge bg={
                          result.customer_prediction.score >= 0.8 ? "primary" :
                          result.customer_prediction.score >= 0.6 ? "warning" :
                          result.customer_prediction.score >= 0.4 ? "secondary" :
                          result.customer_prediction.score >= 0.2 ? "danger" : "dark"
                        }>
                          {result.customer_prediction.score.toFixed(4)}
                        </Badge>
                      </Col>
                    </Row>
                    <Row className="mb-3">
                      <Col sm={3}><strong>Label:</strong></Col>
                      <Col>
                        <Badge bg={
                          result.customer_prediction.score >= 0.8 ? "primary" :
                          result.customer_prediction.score >= 0.6 ? "warning" :
                          result.customer_prediction.score >= 0.4 ? "secondary" :
                          result.customer_prediction.score >= 0.2 ? "danger" : "dark"
                        }>
                          {
                            result.customer_prediction.score >= 0.8 ? "Platinum" :
                            result.customer_prediction.score >= 0.6 ? "Gold" :
                            result.customer_prediction.score >= 0.4 ? "Silver" :
                            result.customer_prediction.score >= 0.2 ? "Bronze" : "Copper"
                          }
                        </Badge>
                      </Col>
                    </Row>
                    <h5 className="mt-4 mb-3">Customer Data:</h5>
                    <div className="table-responsive">
                      <Table striped bordered hover size="sm">
                        <thead>
                          <tr>
                            {result.customer_prediction.data && result.customer_prediction.data[0] &&
                              Object.keys(result.customer_prediction.data[0]).map((key, idx) => (
                                <th key={idx}>{key}</th>
                              ))
                            }
                          </tr>
                        </thead>
                        <tbody>
                          {result.customer_prediction.data &&
                            result.customer_prediction.data.map((row, rowIdx) => (
                              <tr key={rowIdx}>
                                {Object.keys(row).map((key, colIdx) => (
                                  <td key={colIdx}>{row[key]}</td>
                                ))}
                              </tr>
                            ))
                          }
                        </tbody>
                      </Table>
                    </div>
                  </Card.Body>
                </Card>
              </Col>
            </Row>
          )}
        </Card.Body>
      </Card>
    </Container>
  );
}
 
export default App;